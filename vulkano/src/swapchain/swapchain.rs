// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use crossbeam::sync::MsQueue;

use device::Device;
use device::Queue;
use format::Format;
use format::FormatDesc;
use image::sys::Dimensions;
use image::sys::UnsafeImage;
use image::sys::Usage as ImageUsage;
use image::swapchain::SwapchainImage;
use swapchain::CompositeAlpha;
use swapchain::PresentMode;
use swapchain::Surface;
use swapchain::SurfaceTransform;
use sync::Semaphore;
use sync::SharingMode;

use check_errors;
use Error;
use OomError;
use Success;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use vk;

/// Contains the swapping system and the images that can be shown on a surface.
// TODO: #[derive(Debug)] (waiting on https://github.com/aturon/crossbeam/issues/62)
pub struct Swapchain {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: vk::SwapchainKHR,

    /// Pool of semaphores from which a semaphore is retreived when acquiring an image.
    ///
    /// We need to use a queue so that we don't use the same semaphore twice in a row. The length
    /// of the queue is strictly superior to the number of images, in case the driver lets us
    /// acquire an image before it is presented.
    semaphores_pool: MsQueue<Arc<Semaphore>>,

    images_semaphores: Mutex<Vec<Option<Arc<Semaphore>>>>,
}

impl Swapchain {
    /// Builds a new swapchain. Allocates images who content can be made visible on a surface.
    ///
    /// See also the `Surface::get_capabilities` function which returns the values that are
    /// supported by the implementation. All the parameters that you pass to `Swapchain::new`
    /// must be supported. 
    ///
    /// The `clipped` parameter indicates whether the implementation is allowed to discard 
    /// rendering operations that affect regions of the surface which aren't visible. This is
    /// important to take into account if your fragment shader has side-effects or if you want to
    /// read back the content of the image afterwards.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// # Panic
    ///
    /// - Panicks if the device and the surface don't belong to the same instance.
    /// - Panicks if `color_attachment` is false in `usage`.
    ///
    #[inline]
    pub fn new<F, S>(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: F,
                     dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: S,
                     transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                     clipped: bool) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
        where F: FormatDesc, S: Into<SharingMode>
    {
        Swapchain::new_inner(device, surface, num_images, format.format(), dimensions, layers,
                             usage, sharing.into(), transform, alpha, mode, clipped)
    }

    // TODO:
    //pub fn recreate() { ... }

    // TODO: images layouts should always be set to "PRESENT", since we have no way to switch the
    //       layout at present time
    fn new_inner(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: Format,
                 dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: SharingMode,
                 transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                 clipped: bool) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
    {
        // FIXME: check that the parameters are supported

        // FIXME: check that the device and the surface belong to the same instance
        let vk = device.pointers();
        assert!(device.loaded_extensions().khr_swapchain);     // TODO: return error instead

        assert!(usage.color_attachment);
        let usage = usage.to_usage_bits();

        let sharing = sharing.into();

        let swapchain = unsafe {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                SharingMode::Exclusive(id) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (vk::SHARING_MODE_CONCURRENT, ids.len() as u32,
                                                     ids.as_ptr()),
            };

            let infos = vk::SwapchainCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0,   // reserved
                surface: surface.internal_object(),
                minImageCount: num_images,
                imageFormat: format as u32,
                imageColorSpace: vk::COLORSPACE_SRGB_NONLINEAR_KHR,     // only available value
                imageExtent: vk::Extent2D { width: dimensions[0], height: dimensions[1] },
                imageArrayLayers: layers,
                imageUsage: usage,
                imageSharingMode: sh_mode,
                queueFamilyIndexCount: sh_count,
                pQueueFamilyIndices: sh_indices,
                preTransform: transform as u32,
                compositeAlpha: alpha as u32,
                presentMode: mode as u32,
                clipped: if clipped { vk::TRUE } else { vk::FALSE },
                oldSwapchain: 0,      // TODO:
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSwapchainKHR(device.internal_object(), &infos,
                                                    ptr::null(), &mut output)));
            output
        };

        let swapchain = Arc::new(Swapchain {
            device: device.clone(),
            surface: surface.clone(),
            swapchain: swapchain,
            semaphores_pool: MsQueue::new(),
            images_semaphores: Mutex::new(Vec::new()),
        });

        let images = unsafe {
            let mut num = 0;
            try!(check_errors(vk.GetSwapchainImagesKHR(device.internal_object(),
                                                       swapchain.swapchain, &mut num,
                                                       ptr::null_mut())));

            let mut images = Vec::with_capacity(num as usize);
            try!(check_errors(vk.GetSwapchainImagesKHR(device.internal_object(),
                                                       swapchain.swapchain, &mut num,
                                                       images.as_mut_ptr())));
            images.set_len(num as usize);
            images
        };

        let images = images.into_iter().enumerate().map(|(id, image)| unsafe {
            let unsafe_image = UnsafeImage::from_raw(device, image, usage, format,
                                                     Dimensions::Dim2d { width: dimensions[0], height: dimensions[1] }, 1, 1);
            SwapchainImage::from_raw(unsafe_image, format, &swapchain, id as u32).unwrap()     // TODO: propagate error
        }).collect::<Vec<_>>();

        {
            let mut semaphores = swapchain.images_semaphores.lock().unwrap();
            for _ in 0 .. images.len() {
                semaphores.push(None);
            }
        }

        for _ in 0 .. images.len() + 1 {
            swapchain.semaphores_pool.push(try!(Semaphore::new(device)));
        }

        Ok((swapchain, images))
    }

    /// Tries to take ownership of an image in order to draw on it.
    ///
    /// The function returns the index of the image in the array of images that was returned
    /// when creating the swapchain.
    ///
    /// If you try to draw on an image without acquiring it first, the execution will block. (TODO
    /// behavior may change).
    pub fn acquire_next_image(&self, timeout: Duration) -> Result<usize, AcquireError> {
        let vk = self.device.pointers();

        unsafe {
            // TODO: AMD driver crashes when we use the pool
            //let semaphore = self.semaphores_pool.try_pop().expect("Failed to obtain a semaphore from \
            //                                                       the swapchain semaphores pool");
            let semaphore = Semaphore::new(&self.device).unwrap();

            let timeout_ns = timeout.as_secs().saturating_mul(1_000_000_000)
                                              .saturating_add(timeout.subsec_nanos() as u64);

            let mut out = mem::uninitialized();
            let r = try!(check_errors(vk.AcquireNextImageKHR(self.device.internal_object(),
                                                             self.swapchain, timeout_ns,
                                                             semaphore.internal_object(), 0,     // TODO: timeout
                                                             &mut out)));

            let id = match r {
                Success::Success => out as usize,
                Success::Suboptimal => out as usize,        // TODO: give that info to the user
                Success::NotReady => return Err(AcquireError::Timeout),
                Success::Timeout => return Err(AcquireError::Timeout),
                s => panic!("unexpected success value: {:?}", s)
            };

            let mut images_semaphores = self.images_semaphores.lock().unwrap();
            images_semaphores[id] = Some(semaphore);

            Ok(id)
        }
    }

    /// Presents an image on the screen.
    ///
    /// The parameter is the same index as what `acquire_next_image` returned. The image must
    /// have been acquired first.
    ///
    /// The actual behavior depends on the present mode that you passed when creating the
    /// swapchain.
    pub fn present(&self, queue: &Arc<Queue>, index: usize) -> Result<(), OomError> {     // FIXME: wrong error
        let vk = self.device.pointers();

        let wait_semaphore = {
            let mut images_semaphores = self.images_semaphores.lock().unwrap();
            images_semaphores[index].take().expect("Trying to present an image that was \
                                                    not acquired")
        };

        // FIXME: the semaphore will be destroyed ; need to return it

        unsafe {
            let mut result = mem::uninitialized();

            let queue = queue.internal_object_guard();
            let index = index as u32;

            let infos = vk::PresentInfoKHR {
                sType: vk::STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &wait_semaphore.internal_object(),
                swapchainCount: 1,
                pSwapchains: &self.swapchain,
                pImageIndices: &index,
                pResults: &mut result,
            };

            try!(check_errors(vk.QueuePresentKHR(*queue, &infos)));
            //try!(check_errors(result));       // TODO: AMD driver doesn't seem to write the result
        }

        //self.semaphores_pool.push(wait_semaphore);        // TODO: AMD driver crashes when we use the pool
        Ok(())
    }

    /*/// Returns the semaphore that is going to be signalled when the image is going to be ready
    /// to be drawn upon.
    ///
    /// Returns `None` if the image was not acquired first, or was already presented.
    // TODO: racy, as someone could present the image before using the semaphore
    #[inline]
    pub fn image_semaphore(&self, id: u32) -> Option<Arc<Semaphore>> {
        let semaphores = self.images_semaphores.lock().unwrap();
        semaphores[id as usize].as_ref().map(|s| s.clone())
    }*/
    // TODO: the design of this functions depends on https://github.com/KhronosGroup/Vulkan-Docs/issues/155
    #[inline]
    pub fn image_semaphore(&self, id: u32, semaphore: Arc<Semaphore>) -> Option<Arc<Semaphore>> {
        let mut semaphores = self.images_semaphores.lock().unwrap();
        mem::replace(&mut semaphores[id as usize], Some(semaphore))
    }
}

impl Drop for Swapchain {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySwapchainKHR(self.device.internal_object(), self.swapchain, ptr::null());
        }
    }
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AcquireError {
    Timeout,
    SurfaceLost,
    OutOfDate,
}

impl error::Error for AcquireError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            AcquireError::Timeout => "no image is available for acquiring yet",
            AcquireError::SurfaceLost => "the surface of this swapchain is no longer valid",
            AcquireError::OutOfDate => "the swapchain needs to be recreated",
        }
    }
}

impl fmt::Display for AcquireError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for AcquireError {
    #[inline]
    fn from(err: Error) -> AcquireError {
        match err {
            Error::SurfaceLost => AcquireError::SurfaceLost,
            Error::OutOfDate => AcquireError::OutOfDate,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

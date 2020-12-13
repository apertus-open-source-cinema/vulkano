use ::{
    buffer::{
        sys::{BufferCreationError, SparseLevel, UnsafeBuffer},
        usage::BufferUsage,
        BufferAccess, BufferInner,
    },
    check_errors,
    device::{Device, DeviceOwned, Queue},
    image::ImageAccess,
    instance::MemoryType,
    memory::{
        pool::{AllocFromRequirementsFilter, MemoryPoolAlloc},
        DeviceMemory, DeviceMemoryAllocError,
    },
    sync::{AccessError, Sharing},
    OomError, VulkanObject,
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::error;
use std::fmt;
use mmap::{MemoryMap, MapOption};
use std::mem::MaybeUninit;
use std::os::raw::c_int;
use std::ptr;
use std::ops::{Deref, DerefMut};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub struct DmaBufBuffer {
    device: Arc<Device>,
    memory: DeviceMemory,
    fd: c_int,
    inner: UnsafeBuffer,
    access: RwLock<CurrentGpuAccess>,
}

impl DmaBufBuffer {
    pub fn new(
        device: Arc<Device>,
        size: usize,
        usage: BufferUsage,
    ) -> Result<Arc<DmaBufBuffer>, DeviceMemoryAllocError> {
        unsafe {
            let (buffer, mem_reqs) = {
                match UnsafeBuffer::new(
                    device.clone(),
                    size,
                    usage,
                    Sharing::<std::iter::Empty<u32>>::Exclusive,
                    SparseLevel::none(),
                ) {
                    Ok(b) => b,
                    Err(BufferCreationError::AllocError(err)) => return Err(err),
                    Err(_) => unreachable!(),
                }
            };

            let mem_ty = {
                let mut filter = |ty: MemoryType| {
                    if !ty.is_host_visible() {
                        return AllocFromRequirementsFilter::Forbidden;
                    }
                    AllocFromRequirementsFilter::Allowed
                };
                let first_loop = device
                    .physical_device()
                    .memory_types()
                    .map(|t| (t, AllocFromRequirementsFilter::Preferred));
                let second_loop = device
                    .physical_device()
                    .memory_types()
                    .map(|t| (t, AllocFromRequirementsFilter::Allowed));
                first_loop
                    .chain(second_loop)
                    .filter(|&(t, _)| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                    .filter(|&(t, rq)| filter(t) == rq)
                    .next()
                    .expect("Couldn't find a memory type to allocate from")
                    .0
            };

            let size = mem_reqs.size;

            assert!(size >= 1);
            assert_eq!(
                device.physical_device().internal_object(),
                mem_ty.physical_device().internal_object()
            );

            // Note: This check is disabled because MoltenVK doesn't report correct heap sizes yet.
            // This check was re-enabled because Mesa aborts if `size` is Very Large.
            let reported_heap_size = mem_ty.heap().size();
            if reported_heap_size != 0 && size > reported_heap_size {
                return Err(DeviceMemoryAllocError::OomError(
                    OomError::OutOfDeviceMemory,
                ));
            }
            let vk = device.pointers();

            let memory = {
                let physical_device = device.physical_device();
                let mut allocation_count =
                    device.allocation_count().lock().expect("Poisoned mutex");
                if *allocation_count >= physical_device.limits().max_memory_allocation_count() {
                    return Err(DeviceMemoryAllocError::TooManyObjects);
                }
                let vk = device.pointers();

                let export_memory_info = vk::ExportMemoryAllocateInfoKHR {
                    sType: vk::STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                    pNext: ptr::null(),
                    handleType: vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                };

                let dedicated_alloc_info = vk::MemoryDedicatedAllocateInfoKHR {
                    sType: vk::STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
                    pNext: &export_memory_info as *const vk::ExportMemoryAllocateInfoKHR
                        as *const _,
                    image: 0,
                    buffer: buffer.internal_object(),
                };

                let infos = vk::MemoryAllocateInfo {
                    sType: vk::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    pNext: &dedicated_alloc_info as *const vk::MemoryDedicatedAllocateInfoKHR
                        as *const _,
                    allocationSize: size as u64,
                    memoryTypeIndex: mem_ty.id(),
                };

                let mut output = MaybeUninit::uninit();
                check_errors(vk.AllocateMemory(
                    device.internal_object(),
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                *allocation_count += 1;
                output.assume_init()
            };

            let mem = DeviceMemory::raw(memory, device.clone(), size, mem_ty.id());
            buffer.bind_memory(&mem, 0)?;

            let get_fd_info = vk::MemoryGetFdInfoKHR {
                sType: vk::STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                pNext: ptr::null(),
                memory: memory,
                handleType: vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            };

            let mut fd = MaybeUninit::uninit();
            check_errors(vk.GetMemoryFdKHR(
                device.internal_object(),
                &get_fd_info as *const vk::MemoryGetFdInfoKHR,
                fd.as_mut_ptr(),
            ))?;
            let fd = fd.assume_init();

            // println!("got fd {}", fd);

            Ok(Arc::new(DmaBufBuffer {
                fd: fd,
                device: device,
                inner: buffer,
                memory: mem,
                access: RwLock::new(CurrentGpuAccess::NonExclusive {
                    num: AtomicUsize::new(0),
                }),
            }))
        }
    }

    pub fn read(&self) -> Result<ReadLock, ReadLockError> {
        let lock = match self.access.try_read() {
            Some(l) => l,
            // TODO: if a user simultaneously calls .write(), and write() is currently finding out
            //       that the buffer is in fact GPU locked, then we will return a CpuWriteLocked
            //       error instead of a GpuWriteLocked ; is this a problem? how do we fix this?
            None => return Err(ReadLockError::CpuWriteLocked),
        };

        if let CurrentGpuAccess::Exclusive { .. } = *lock {
            return Err(ReadLockError::GpuWriteLocked);
        }

        Ok(ReadLock {
            inner: unsafe { MemoryMap::new(self.inner.size(), &[MapOption::MapFd(self.fd), MapOption::MapReadable, MapOption::MapNonStandardFlags(0x0001)]).unwrap() },
            lock: lock,
        })
    }

    pub fn write(&self) -> Result<WriteLock, WriteLockError> {
        let lock = match self.access.try_write() {
            Some(l) => l,
            // TODO: if a user simultaneously calls .read() or .write(), and the function is
            //       currently finding out that the buffer is in fact GPU locked, then we will
            //       return a CpuLocked error instead of a GpuLocked ; is this a problem?
            //       how do we fix this?
            None => return Err(WriteLockError::CpuLocked),
        };

        match *lock {
            CurrentGpuAccess::NonExclusive { ref num } if num.load(Ordering::SeqCst) == 0 => (),
            _ => return Err(WriteLockError::GpuLocked),
        }

        Ok(WriteLock {
            inner: unsafe {  MemoryMap::new(self.inner.size(), &[MapOption::MapFd(self.fd), MapOption::MapReadable, MapOption::MapWritable, MapOption::MapNonStandardFlags(0x0001)]).unwrap() },
            lock: lock,
        })
    }
}

#[derive(Debug)]
enum CurrentGpuAccess {
    NonExclusive {
        // Number of non-exclusive GPU accesses. Can be 0.
        num: AtomicUsize,
    },
    Exclusive {
        // Number of exclusive locks. Cannot be 0. If 0 is reached, we must jump to `NonExclusive`.
        num: usize,
    },
}

unsafe impl DeviceOwned for DmaBufBuffer {
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl BufferAccess for DmaBufBuffer {
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        self.conflict_key() == other.conflict_key()
    }

    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        false
    }

    fn conflict_key(&self) -> (u64, usize) {
        (self.inner.key(), 0)
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, _: &Queue) -> Result<(), AccessError> {
        if exclusive_access {
            let mut lock = match self.access.try_write() {
                Some(lock) => lock,
                None => return Err(AccessError::AlreadyInUse),
            };

            match *lock {
                CurrentGpuAccess::NonExclusive { ref num } if num.load(Ordering::SeqCst) == 0 => (),
                _ => return Err(AccessError::AlreadyInUse),
            };

            *lock = CurrentGpuAccess::Exclusive { num: 1 };
            Ok(())
        } else {
            let lock = match self.access.try_read() {
                Some(lock) => lock,
                None => return Err(AccessError::AlreadyInUse),
            };

            match *lock {
                CurrentGpuAccess::Exclusive { .. } => return Err(AccessError::AlreadyInUse),
                CurrentGpuAccess::NonExclusive { ref num } => num.fetch_add(1, Ordering::SeqCst),
            };

            Ok(())
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        // First, handle if we have a non-exclusive access.
        {
            // Since the buffer is in use by the GPU, it is invalid to hold a write-lock to
            // the buffer. The buffer can still be briefly in a write-locked state for the duration
            // of the check though.
            let read_lock = self.access.read();
            if let CurrentGpuAccess::NonExclusive { ref num } = *read_lock {
                let prev = num.fetch_add(1, Ordering::SeqCst);
                debug_assert!(prev >= 1);
                return;
            }
        }

        // If we reach here, this means that `access` contains `CurrentGpuAccess::Exclusive`.
        {
            // Same remark as above, but for writing.
            let mut write_lock = self.access.write();
            if let CurrentGpuAccess::Exclusive { ref mut num } = *write_lock {
                *num += 1;
            } else {
                unreachable!()
            }
        }
    }

    #[inline]
    unsafe fn unlock(&self) {
        // First, handle if we had a non-exclusive access.
        {
            // Since the buffer is in use by the GPU, it is invalid to hold a write-lock to
            // the buffer. The buffer can still be briefly in a write-locked state for the duration
            // of the check though.
            let read_lock = self.access.read();
            if let CurrentGpuAccess::NonExclusive { ref num } = *read_lock {
                let prev = num.fetch_sub(1, Ordering::SeqCst);
                debug_assert!(prev >= 1);
                return;
            }
        }

        // If we reach here, this means that `access` contains `CurrentGpuAccess::Exclusive`.
        {
            // Same remark as above, but for writing.
            let mut write_lock = self.access.write();
            if let CurrentGpuAccess::Exclusive { ref mut num } = *write_lock {
                if *num != 1 {
                    *num -= 1;
                    return;
                }
            } else {
                // Can happen if we lock in exclusive mode N times, and unlock N+1 times with the
                // last two unlocks happen simultaneously.
                panic!()
            }

            *write_lock = CurrentGpuAccess::NonExclusive {
                num: AtomicUsize::new(0),
            };
        }
    }
}

pub struct ReadLock<'a> {
    inner: MemoryMap,
    lock: RwLockReadGuard<'a, CurrentGpuAccess>,
}

impl<'a> Deref for ReadLock<'a> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.inner.data(), self.inner.len()) }
    }
}

/// Error when attempting to CPU-read a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadLockError {
    /// The buffer is already locked for write mode by the CPU.
    CpuWriteLocked,
    /// The buffer is already locked for write mode by the GPU.
    GpuWriteLocked,
}

impl error::Error for ReadLockError {}

impl fmt::Display for ReadLockError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", match *self {
            ReadLockError::CpuWriteLocked => {
                "the buffer is already locked for write mode by the CPU"
            }
            ReadLockError::GpuWriteLocked => {
                "the buffer is already locked for write mode by the GPU"
            }
        })
    }
}

/// Object that can be used to read or write the content of a `CpuAccessibleBuffer`.
///
/// Note that this object holds a rwlock write guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct WriteLock<'a> {
    inner: MemoryMap,
    lock: RwLockWriteGuard<'a, CurrentGpuAccess>,
}

impl<'a> Deref for WriteLock<'a> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.inner.data(), self.inner.len()) }
    }
}

impl<'a> DerefMut for WriteLock<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.inner.data(), self.inner.len()) }
    }
}

/// Error when attempting to CPU-write a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteLockError {
    /// The buffer is already locked by the CPU.
    CpuLocked,
    /// The buffer is already locked by the GPU.
    GpuLocked,
}

impl error::Error for WriteLockError {}

impl fmt::Display for WriteLockError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", match *self {
            WriteLockError::CpuLocked => "the buffer is already locked by the CPU",
            WriteLockError::GpuLocked => "the buffer is already locked by the GPU",
        })
    }
}

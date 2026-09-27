//! Explicit standalone oneMKL 2024.2 F32 FFT for the pinned MiMo mel component.
//!
//! This loads `libmkl_rt.so.2` from a caller-supplied path after checking
//! MKL_THREADING_LAYER=SEQUENTIAL. It does not load Torch or search the host
//! for a library, and it does not enable audio serving.

use libloading::Library;
use std::ffi::{CStr, OsStr, c_char, c_void};
use std::path::Path;

use crate::mimo_audio_pcm_mel::FFT_SIZE;

const FFT_BINS: usize = FFT_SIZE / 2 + 1;
const DFTI_SINGLE: i32 = 35;
const DFTI_REAL: i32 = 33;
const DFTI_PLACEMENT: i32 = 11;
const DFTI_NOT_INPLACE: i32 = 44;
const DFTI_CONJUGATE_EVEN_STORAGE: i32 = 10;
const DFTI_COMPLEX_COMPLEX: i32 = 39;
const DFTI_THREAD_LIMIT: i32 = 27;
const PINNED_MKL_BUILD: &str = "Version 2024.2-Product Build 20240605";

type Create = unsafe extern "C" fn(*mut *mut c_void, i32, i32, i64, ...) -> i64;
type SetValue = unsafe extern "C" fn(*mut c_void, i32, ...) -> i64;
type Commit = unsafe extern "C" fn(*mut c_void) -> i64;
type Compute = unsafe extern "C" fn(*mut c_void, *const f32, ...) -> i64;
type Free = unsafe extern "C" fn(*mut *mut c_void) -> i64;
type Version = unsafe extern "C" fn(*mut c_char, i32);

pub struct OneMklFft960 {
    _library: Library,
    descriptor: *mut c_void,
    compute: Compute,
    free: Free,
}

impl OneMklFft960 {
    /// Load the standalone Intel oneMKL build used by the pinned Torch 2.6
    /// CPU wheel. The caller owns packaging and the library's native
    /// dependencies. This component does not load `libtorch_cpu.so`.
    pub fn open(path: &Path) -> Result<Self, String> {
        if !path.is_absolute() || path.file_name() != Some(OsStr::new("libmkl_rt.so.2")) {
            return Err("MiMo FFT needs an absolute standalone libmkl_rt.so.2 path".into());
        }
        if std::env::var_os("MKL_THREADING_LAYER").as_deref() != Some(OsStr::new("SEQUENTIAL")) {
            return Err(
                "MiMo FFT requires MKL_THREADING_LAYER=SEQUENTIAL before loading oneMKL".into(),
            );
        }
        // SAFETY: Symbol signatures and DFTI enum values follow mkl_dfti.h.
        // The Library remains owned by the returned object until after its
        // descriptor has been freed in Drop.
        unsafe {
            let library = Library::new(path).map_err(|error| format!("load oneMKL: {error}"))?;
            let version: Version = *library
                .get(b"mkl_get_version_string\0")
                .map_err(|error| format!("oneMKL version symbol: {error}"))?;
            let mut buffer = [0_i8; 256];
            version(buffer.as_mut_ptr(), buffer.len() as i32);
            let found = CStr::from_ptr(buffer.as_ptr()).to_string_lossy();
            if !found.contains(PINNED_MKL_BUILD) {
                return Err(format!(
                    "MiMo FFT needs oneMKL {PINNED_MKL_BUILD}, got {found}"
                ));
            }
            let create: Create = *library
                .get(b"DftiCreateDescriptor\0")
                .map_err(|error| format!("oneMKL create symbol: {error}"))?;
            let set_value: SetValue = *library
                .get(b"DftiSetValue\0")
                .map_err(|error| format!("oneMKL set symbol: {error}"))?;
            let commit: Commit = *library
                .get(b"DftiCommitDescriptor\0")
                .map_err(|error| format!("oneMKL commit symbol: {error}"))?;
            let compute: Compute = *library
                .get(b"DftiComputeForward\0")
                .map_err(|error| format!("oneMKL compute symbol: {error}"))?;
            let free: Free = *library
                .get(b"DftiFreeDescriptor\0")
                .map_err(|error| format!("oneMKL free symbol: {error}"))?;

            let mut descriptor = std::ptr::null_mut();
            let status = create(
                &mut descriptor,
                DFTI_SINGLE,
                DFTI_REAL,
                1_i64,
                FFT_SIZE as i64,
            );
            if status != 0 || descriptor.is_null() {
                if !descriptor.is_null() {
                    free(&mut descriptor);
                }
                return Err(format!("oneMKL 960-point descriptor failed: {status}"));
            }
            let fft = Self {
                _library: library,
                descriptor,
                compute,
                free,
            };
            let status = set_value(fft.descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            if status != 0 {
                return Err(format!("oneMKL out-of-place setting failed: {status}"));
            }
            let status = set_value(
                fft.descriptor,
                DFTI_CONJUGATE_EVEN_STORAGE,
                DFTI_COMPLEX_COMPLEX,
            );
            if status != 0 {
                return Err(format!("oneMKL complex output setting failed: {status}"));
            }
            let status = set_value(fft.descriptor, DFTI_THREAD_LIMIT, 1_i32);
            if status != 0 {
                return Err(format!("oneMKL single-thread setting failed: {status}"));
            }
            let status = commit(fft.descriptor);
            if status != 0 {
                return Err(format!("oneMKL descriptor commit failed: {status}"));
            }
            Ok(fft)
        }
    }

    /// `windowed` is one already windowed F32 frame. `magnitudes` receives
    /// the unnormalized one-sided 481-bin complex modulus.
    pub fn magnitudes(&mut self, windowed: &[f32], magnitudes: &mut [f32]) -> Result<(), String> {
        if windowed.len() != FFT_SIZE || magnitudes.len() != FFT_BINS {
            return Err("MiMo oneMKL FFT needs 960 F32 samples and 481 magnitudes".into());
        }
        let mut complex = [0.0_f32; FFT_BINS * 2];
        // SAFETY: The committed descriptor is a 960-point, out-of-place F32
        // real-to-complex transform. Both input and output buffers have the
        // required extents and live through DftiComputeForward.
        let status =
            unsafe { (self.compute)(self.descriptor, windowed.as_ptr(), complex.as_mut_ptr()) };
        if status != 0 {
            return Err(format!("oneMKL FFT failed: {status}"));
        }
        for (bin, magnitude) in magnitudes.iter_mut().enumerate() {
            *magnitude = complex[2 * bin].hypot(complex[2 * bin + 1]);
        }
        Ok(())
    }
}

impl Drop for OneMklFft960 {
    fn drop(&mut self) {
        if !self.descriptor.is_null() {
            // SAFETY: The descriptor was created by the loaded oneMKL
            // library and is freed before that library is dropped.
            unsafe {
                (self.free)(&mut self.descriptor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_implicit_lookup_and_torch_library() {
        assert!(OneMklFft960::open(Path::new("libmkl_rt.so.2")).is_err());
        assert!(OneMklFft960::open(Path::new("/tmp/libtorch_cpu.so")).is_err());
    }
}

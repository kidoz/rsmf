#[cfg(feature = "ort-api-23-allocator-stats")]
use crate::session::RuntimeAllocatorStat;
use crate::session::RuntimeAllocatorStats;

#[cfg(feature = "ort-api-23-allocator-stats")]
use ort::AsPointer;
use ort::memory::Allocator;
#[cfg(feature = "ort-api-23-allocator-stats")]
use std::ffi::CStr;
#[cfg(feature = "ort-api-23-allocator-stats")]
use std::ptr::{self, NonNull};

#[cfg(feature = "ort-api-23-allocator-stats")]
struct KeyValuePairsGuard(NonNull<ort::sys::OrtKeyValuePairs>);

#[cfg(feature = "ort-api-23-allocator-stats")]
impl Drop for KeyValuePairsGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a non-null pointer returned by ORT
        // `AllocatorGetStats`; ORT requires callers to release the returned
        // key/value object with `ReleaseKeyValuePairs` after copying values.
        unsafe {
            (ort::api().ReleaseKeyValuePairs)(self.0.as_ptr());
        }
    }
}

#[cfg(feature = "ort-api-23-allocator-stats")]
pub(crate) fn provider_allocator_stats(allocator: &Allocator) -> RuntimeAllocatorStats {
    let mut stats_ptr: *mut ort::sys::OrtKeyValuePairs = ptr::null_mut();
    // SAFETY: `allocator.ptr()` is a valid non-null ORT allocator pointer for
    // the borrowed safe `Allocator`, and `stats_ptr` is an out pointer owned by
    // this function until released by `KeyValuePairsGuard`.
    let status = unsafe { (ort::api().AllocatorGetStats)(allocator.ptr(), &mut stats_ptr) };
    if let Err(error) = unsafe { ort::Error::result_from_status(status) } {
        return RuntimeAllocatorStats::unavailable(format!(
            "ORT allocator stats query failed: {error}"
        ));
    }

    let Some(stats_ptr) = NonNull::new(stats_ptr) else {
        return RuntimeAllocatorStats::unavailable("ORT allocator returned no stats object");
    };
    let _guard = KeyValuePairsGuard(stats_ptr);

    let mut keys: *const *const std::ffi::c_char = ptr::null();
    let mut values: *const *const std::ffi::c_char = ptr::null();
    let mut count = 0usize;
    // SAFETY: `stats_ptr` remains alive for this scope through `_guard`. ORT
    // writes arrays of borrowed C string pointers valid until release.
    unsafe {
        (ort::api().GetKeyValuePairs)(stats_ptr.as_ptr(), &mut keys, &mut values, &mut count);
    }

    if count == 0 {
        return RuntimeAllocatorStats::from_entries(Vec::new());
    }
    if keys.is_null() || values.is_null() {
        return RuntimeAllocatorStats::unavailable(
            "ORT allocator stats returned null key/value arrays",
        );
    }

    // SAFETY: ORT returned non-null arrays with `count` entries. Individual
    // null entries are handled below without dereferencing.
    let keys = unsafe { std::slice::from_raw_parts(keys, count) };
    let values = unsafe { std::slice::from_raw_parts(values, count) };
    let mut entries = Vec::with_capacity(count);
    for (key, value) in keys.iter().zip(values.iter()) {
        if key.is_null() || value.is_null() {
            return RuntimeAllocatorStats::unavailable(
                "ORT allocator stats contained a null key or value",
            );
        }
        // SAFETY: ORT key/value entries are null-terminated C strings valid
        // until `ReleaseKeyValuePairs`.
        let key = unsafe { CStr::from_ptr(*key) }
            .to_string_lossy()
            .into_owned();
        let value = unsafe { CStr::from_ptr(*value) }
            .to_string_lossy()
            .into_owned();
        entries.push(RuntimeAllocatorStat { key, value });
    }

    RuntimeAllocatorStats::from_entries(entries)
}

#[cfg(not(feature = "ort-api-23-allocator-stats"))]
pub(crate) fn provider_allocator_stats(_allocator: &Allocator) -> RuntimeAllocatorStats {
    RuntimeAllocatorStats::unavailable(
        "rsmf-runtime was built without the ort-api-23-allocator-stats feature",
    )
}

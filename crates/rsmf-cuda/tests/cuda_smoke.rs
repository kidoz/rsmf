#[cfg(target_os = "linux")]
#[test]
fn test_cuda_compilation() {
    // Just ensure the crate is usable.
    let _ = rsmf_cuda::detect_capabilities();
}

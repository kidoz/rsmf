#[test]
fn test_wgpu_compilation() {
    // Just ensure the crate is usable.
    let _ = rsmf_wgpu::detect_capabilities();
}

#[test]
fn wgpu_linear_matches_reference_when_adapter_is_available() {
    let Ok(executor) = rsmf_wgpu::WgpuLinearExecutor::new() else {
        return;
    };
    let output = executor
        .linear(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            3,
        )
        .unwrap();

    assert_eq!(output, vec![17.0, 23.0, 29.0, 39.0, 53.0, 67.0]);
}

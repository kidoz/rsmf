use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[test]
fn test_pack_npy_smoke() {
    // 1. Create a dummy .npy file
    let mut npy = NamedTempFile::new().unwrap();
    // Simple NPY header + data for 4 floats [0.0, 1.0, 2.0, 3.0]
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }";
    let mut header_bytes = header.as_bytes().to_vec();
    while (header_bytes.len() + 10) % 64 != 63 {
        header_bytes.push(b' ');
    }
    header_bytes.push(b'\n');

    let mut file_bytes = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    file_bytes.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
    file_bytes.extend_from_slice(&header_bytes);
    for i in 0..4 {
        file_bytes.extend_from_slice(&(i as f32).to_le_bytes());
    }
    npy.write_all(&file_bytes).unwrap();

    let out = NamedTempFile::new().unwrap();

    // 2. Run rsmf pack
    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "rsmf-cli",
            "--",
            "pack",
            "--from-npy",
            npy.path().to_str().unwrap(),
            "--out",
            out.path().to_str().unwrap(),
        ])
        .status()
        .expect("failed to execute process");

    assert!(status.success());

    // 3. Verify with rsmf inspect
    let inspect = Command::new("cargo")
        .args([
            "run",
            "-p",
            "rsmf-cli",
            "--",
            "inspect",
            out.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(inspect.status.success());
    let stdout = String::from_utf8(inspect.stdout).unwrap();
    assert!(stdout.contains("Tensors:  1"));
}

//! WGPU upload benchmark. Only built with `--features wgpu`. Skips gracefully
//! when no adapter is available.

#![cfg(feature = "wgpu")]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsmf_bench::{DEFAULT_COLS, DEFAULT_ROWS, build_fixture};
use rsmf_core::RsmfFile;
use rsmf_wgpu::{UploadOptions, detect_capabilities, request_device, upload_canonical_tensor};
use tempfile::tempdir;

fn bench_wgpu_upload(c: &mut Criterion) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fixture.rsmf");
    build_fixture(&path, DEFAULT_ROWS, DEFAULT_COLS).expect("build fixture");
    let file = RsmfFile::open(&path).expect("open");

    let caps = match detect_capabilities() {
        Some(c) => c,
        None => {
            eprintln!("skipping wgpu_upload bench: no adapter available");
            return;
        }
    };
    let device = match request_device(&caps) {
        Some(d) => d,
        None => {
            eprintln!("skipping wgpu_upload bench: could not request device");
            return;
        }
    };

    c.bench_function("rsmf_wgpu_upload", |b| {
        b.iter(|| {
            let view = file.tensor_view("weight").expect("view");
            let buffer = upload_canonical_tensor(&device, view.bytes(), &UploadOptions::default())
                .expect("upload");
            black_box(buffer);
        });
    });
}

criterion_group!(benches, bench_wgpu_upload);
criterion_main!(benches);

//! Tensor access latency: time to resolve a tensor view and read the first
//! few bytes.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsmf_bench::{DEFAULT_COLS, DEFAULT_ROWS, build_fixture};
use rsmf_core::RsmfFile;
use tempfile::tempdir;

fn bench_tensor_access(c: &mut Criterion) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fixture.rsmf");
    build_fixture(&path, DEFAULT_ROWS, DEFAULT_COLS).expect("build fixture");
    let file = RsmfFile::open(&path).expect("open");

    c.bench_function("rsmf_tensor_view_first_byte", |b| {
        b.iter(|| {
            let view = file.tensor_view("weight").expect("view");
            black_box(view.bytes()[0]);
        });
    });
}

criterion_group!(benches, bench_tensor_access);
criterion_main!(benches);

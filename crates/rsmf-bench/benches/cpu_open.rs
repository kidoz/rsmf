//! CPU open / structural-validation benchmark.
//!
//! Opens a small RSMF fixture repeatedly and measures open-time latency
//! including structural validation.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsmf_bench::{DEFAULT_COLS, DEFAULT_ROWS, build_fixture};
use rsmf_core::RsmfFile;
use tempfile::tempdir;

fn bench_cpu_open(c: &mut Criterion) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fixture.rsmf");
    build_fixture(&path, DEFAULT_ROWS, DEFAULT_COLS).expect("build fixture");

    c.bench_function("rsmf_open", |b| {
        b.iter(|| {
            let f = RsmfFile::open(&path).expect("open");
            black_box(f.inspect());
        });
    });
}

criterion_group!(benches, bench_cpu_open);
criterion_main!(benches);

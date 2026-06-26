//! MoE dispatch grouping throughput.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rsmf_moe_runtime::batch_by_destination;

fn bench_moe_dispatch(c: &mut Criterion) {
    const TOKENS: usize = 4096;
    const EXPERTS: u32 = 8;

    let assignments: Vec<u32> = (0..TOKENS)
        .map(|token| ((token * 1103515245 + 12345) as u32) % EXPERTS)
        .collect();

    let mut group = c.benchmark_group("moe_dispatch");
    group.throughput(Throughput::Elements(TOKENS as u64));
    group.bench_function("batch_by_destination", |b| {
        b.iter(|| {
            let batches = batch_by_destination(black_box(&assignments));
            black_box(batches);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_moe_dispatch);
criterion_main!(benches);

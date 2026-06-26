# RSMF — Minimal MoE Runtime PoC

`rsmf-moe-runtime` is a proof-of-concept runtime. It is not a
general inference engine and is excluded from `default-members`.

## Pipeline

For one MoE layer, the runtime:

1. Reads `moe.*` metadata to find the layer router and expert `up` / `down`
   tensors.
2. Runs host-side top-1 gating with a router tensor shaped
   `[n_experts, input_width]`.
3. Batches tokens by destination expert.
4. Resolves each expert's `shard_id` through `PlacementManifest`.
5. Loads expert weights through the normal `RsmfFile` sharded tensor path.
6. Runs F32 expert matmuls and scatters rows back into token order.

The reference path uses the same router and expert math without placement-based
batching. Tests compare the two paths on a fixed 2-shard fixture.

## WGPU Execution And Fallback

The optional `wgpu` feature runs expert `up` / `down` matmuls through a small
WGPU compute shader when an adapter is available. A single physical adapter may
back multiple logical WGPU placement devices in this PoC; the placement records
are still used for routing and reporting.

If the build lacks the feature, or no adapter is available, the runtime reports
`RuntimeBackend::CpuFallback` and continues on CPU. This keeps CI hardware-free
while preserving the placement and routing contract.

## Timing

`MoeRunReport` records:

- `gating_time`
- `dispatch_time`
- `compute_time`
- `combine_time`
- `tokens_per_second()`

The Criterion `moe_dispatch` bench isolates token batching throughput. Use it
with:

```sh
cargo bench -p rsmf-bench --bench moe_dispatch
```

# RSMF — MoE Runtime

`rsmf-moe-runtime` is the optional expert-parallel runtime foundation. It is
excluded from `default-members` so default workspace builds remain
hardware-free, but the crate exposes production contracts for placement
planning, resident layer preparation, deterministic routing, and reference
comparison.

## Pipeline

For one MoE layer, the runtime:

1. Reads `moe.*` metadata to find the layer router and expert `up` / `down`
   tensors.
2. Runs host-side routing with a router tensor shaped
   `[n_experts, input_width]`. `run_layer_top1` remains the compatibility
   path; `run_layer_routed` uses `MoeRuntimeOptions::routing` for top-1 or
   top-k routing.
3. Batches tokens by destination expert.
4. Resolves each expert's `shard_id` through `PlacementManifest`.
5. Loads expert weights through the normal `RsmfFile` sharded tensor path.
6. Prepares a resident `MoeLayerPlan` with decoded router/expert weights,
   per-device resident byte accounting, placement capacity checks, and
   prefetch-group summaries.
7. Runs F32 expert matmuls by placement device and scatters weighted rows back
   into token order.

The reference path uses the same router and expert math without placement-based
batching. Tests compare the two paths on a fixed 2-shard fixture.

`MoeRuntime::run_layer_top1_checked` and
`MoeRuntime::run_prepared_layer_top1_checked` execute the routed path, execute
the single-device reference path, report maximum absolute difference, and record
the measured routed/reference wall-time ratio.

## Runtime Contract

The runtime is intentionally stricter than the metadata index:

- `run_layer_top1` only accepts `moe.top_k=1` when `moe.top_k` is present.
- `run_layer_routed` accepts `MoeRoutingPolicy::TopK` when `moe.top_k` is
  absent or matches the requested `k`; selected experts are combined with either
  stable-softmax weights or equal `1.0` weights.
- A layer must have exactly one router tensor, and router tensors must not set
  `moe.expert`.
- Each router row must have a matching expert id with exactly one `up` and one
  `down` tensor.
- `SiluGated` activation requires exactly one `gate` tensor per expert.
- Expert `up` / `down` / `gate` tensors must live on the same `shard_id` so a
  placement record names one owning device for the expert.
- Placement device capacity is enforced when `capacity_bytes` is non-zero.
- Shared MoE experts are rejected by this crate; the runtime covers routed
  expert-sharded layers where each expert is resident on one primary shard.

`MoeRuntimeOptions::limits` rejects oversized token batches, decoded rank-2
tensors, prepared-layer device/expert counts, per-device resident bytes, and
output buffers before the runtime allocates large working memory. The defaults
are deliberately finite and can be widened or disabled per field by callers
that own the deployment envelope.

Prepared plans also report:

- `MoeTransferPlan`, an observability contract for host/device transfer intent
  based on placement device kind and memory tier,
- `MultiAdapterStatus`, which distinguishes CPU-only execution, available WGPU
  adapter coverage, partial adapter coverage, and logical single-adapter
  fallback,
- `MoeCollectivePlan`, currently empty because expert-sharded execution does
  not require tensor-parallel collectives.

`CpuCollectives` provides hermetic reference `all_reduce_sum`, `gather`, and
reported `execute` helpers for future tensor-parallel backend validation.

## WGPU Execution And Fallback

The optional `wgpu` feature runs expert `up` / `down` matmuls through a small
WGPU compute shader when an adapter is available. The shader pipeline and
resident weight-buffer cache are created once per physical adapter executor and
reused across expert batches. Runtime construction enumerates physical WGPU
adapters, creates an executor pool up to the number of logical WGPU placement
devices, and assigns logical placement device ids to physical executors. If
fewer physical adapters are available than logical placement devices, multiple
logical devices share an executor and the plan reports
`MultiAdapterStatus::Partial` or `MultiAdapterStatus::LogicalSingleAdapter`,
depending on how many physical executors are active.

WGPU placement-device groups are dispatched in scoped host threads per physical
executor slot. Each slot processes its assigned logical devices sequentially,
then the host combines per-device outputs after the scoped threads join.

Resident WGPU matrix cache misses are reported as executable
`HostToDevice` transfer stages in each `DeviceRunReport::transfer`: cache
misses upload decoded resident weight bytes and record elapsed upload time,
while cache hits report zero transfer bytes. CPU/RAM execution reports
`MoeTransferKind::None` with zero transfer bytes.

Transfer support is routed through an internal transfer executor boundary. The
current executors support CPU/RAM no-op transfers and WGPU host-to-device
resident weight uploads. Device-to-host and peer-to-peer movement remain typed
unsupported operations until real backend copy paths exist.

Mixed placement is supported at the batch boundary: WGPU placement devices use
the executor pool, while unmapped/CPU placement devices fall back to the CPU
expert path.

If the build lacks the feature, or no adapter is available, the runtime reports
`RuntimeBackend::CpuFallback` and continues on CPU. This keeps CI hardware-free
while preserving the placement and routing contract.

## Timing

`MoeRunReport` records:

- prepared `plan` residency and placement details
- per-run `device_batches`
- per-device `device_runs`
- per-device WGPU `weight_cache_hits` / `weight_cache_misses`
- per-device executed `transfer` kind, bytes, duration, and cache events
- per-run `collective_runs`, currently empty for expert-sharded execution and
  populated by CPU reference collective helpers
- `gating_time`
- `dispatch_time`
- `compute_time`
- `combine_time`
- `tokens_per_second()`

CPU reference tensor-parallel collectives are available through
`CpuCollectives::execute`. Device-backed tensor-parallel collectives are not
implemented yet. Prepared layer plans report
`TensorParallelismStatus::NotRequired` for the current expert-sharded contract
where each expert is owned by one shard/device. Future tensor-sliced experts
must report an explicit unavailable status until real device collectives exist.

The Criterion `moe_dispatch` bench isolates token batching throughput and also
includes a tiny end-to-end routed top-k fixture. Use it with:

```sh
cargo bench -p rsmf-bench --bench moe_dispatch
```

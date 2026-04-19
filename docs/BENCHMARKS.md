# RSMF — Benchmarks

All benchmarks live in `crates/rsmf-bench`. The crate uses
[criterion](https://github.com/bheisler/criterion.rs) and ships synthetic
fixtures (no external model downloads).

## Running

CPU benchmarks, laptop-sized synthetic fixture:

```sh
cargo bench -p rsmf-bench
```

Optional GPU upload bench. Requires an available GPU adapter; skips gracefully
when none is present:

```sh
cargo bench -p rsmf-bench --features wgpu
```

Compare across runs with criterion's built-in baseline diffs:

```sh
cargo bench -p rsmf-bench -- --save-baseline before
# ...make changes...
cargo bench -p rsmf-bench -- --baseline before
```

## Fixture

Every bench calls `rsmf_bench::build_fixture(path, DEFAULT_ROWS, DEFAULT_COLS)`
which writes a single 512 × 512 F32 tensor (1 MiB of payload) into a fresh
tempdir. Row-major, canonical variant only, no compression, no packed variants.

## What each bench measures

### `cpu_open`

Opens the fixture through `RsmfFile::open` and walks the manifest via
`file.inspect()`. Timing covers:

- `mmap` of the master file,
- preamble + section-table decode + validation,
- manifest decode,
- cross-reference validation,
- the `ManifestSummary` build performed by `inspect()`.

Source: `crates/rsmf-bench/benches/cpu_open.rs`.

### `tensor_access`

Opens the fixture **once** outside the timing loop, then inside the loop:

- looks up `"weight"` by name,
- materializes a `TensorView` over the mmap,
- reads the first byte (`view.bytes()[0]`) under `black_box`.

This isolates name → descriptor lookup + slice construction; it does **not**
include open, decode, or decompression cost.

Source: `crates/rsmf-bench/benches/tensor_access.rs`.

### `wgpu_upload` (feature `wgpu`)

Opens the same fixture and, inside the timing loop:

- resolves `tensor_view("weight")`,
- calls `rsmf_wgpu::upload_canonical_tensor(device, view.bytes(), &UploadOptions::default())`.

The default `UploadOptions::chunk_bytes` is 4 MiB (`crates/rsmf-wgpu/src/upload.rs`),
so the 1 MiB fixture fits in a single `queue.write_buffer` call. If no adapter
is available the bench prints `skipping wgpu_upload bench: …` to stderr and
returns without failing the run.

Source: `crates/rsmf-bench/benches/wgpu_upload.rs`.

## Methodology

- **Cache effects:** the fixture is 1 MiB. On M-series Apple Silicon it
  comfortably fits in L2. On x86 cores with a 512 KiB or 1 MiB private L2 it
  does **not** — expect the first few iterations to spill to L3 before the
  working set stabilises.
- **Isolation:** every bench rebuilds the fixture in a unique `tempdir`; no
  user paths are touched.
- **GPU skip behaviour:** the WGPU bench short-circuits with a stderr message
  when no adapter / device is available, so CI runs without GPUs still pass.

## Reference numbers

The numbers below are **illustrative only** — re-run the benches on your
hardware for ground truth. A healthy warm-cache run on a 2024-era laptop lands
in this envelope:

```
rsmf_open                 time:   [120 µs  125 µs  130 µs]
rsmf_tensor_view_first_byte
                          time:   [ 45 ns   47 ns   49 ns]
```

When filing performance regressions, include the CPU model, OS, `rustc
--version`, and the full criterion output (including the `change:` line if you
used `--save-baseline`).

## Not benchmarked in-tree

- **safetensors baseline.** A fair comparison needs separate harnessing —
  safetensors' read path is implemented differently (on-demand JSON header
  deserialisation plus zero-copy slices) so a meaningful latency number must
  control for tensor layout and access pattern. Not in scope for the MVP.
- **rsmf-python latency.** PyO3 call overhead and `decode_f32` copy cost in
  the Python bindings (`crates/rsmf-python`) aren't measured here. Out of
  scope until the Python surface stabilises.
- **Large fixtures.** The bench crate ships a single 1 MiB synthetic tensor.
  To exercise the multi-variant / compressed / sharded paths, pack a larger
  real model with the `rsmf` CLI and measure `RsmfFile::open` + `tensor_view`
  directly from your own harness.


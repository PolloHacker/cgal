# WNNC normal estimation — C++ implementation

A standalone, modern C++20 port of the WNNC iterative normal estimation
method, with no PyTorch or CUDA dependency. Parallelism is plain
`std::thread`; the only bundled third-party code is the
[CLI11](https://github.com/CLIUtils/CLI11) command-line parser.

The same core also powers the torch-free Python package: `pip install
../python` builds a pybind11 binding (`python/src/wnnc_bindings.cpp`) around
these headers, used by `python/main_wnnc.py`.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # optional: run the test suite
```

## Usage

Mirrors the Python driver:

```bash
# clean, uniform samples
./build/wnnc ../data/Armadillo_40000.xyz --width_config l0 --progress

# noisy or non-uniform points: l1 (small noise) ... l5 (large noise)
./build/wnnc ../data/bunny_noised.xyz --width_config l5 --progress

# custom smoothing widths
./build/wnnc cloud.xyz --width_config custom --wsmin 0.03 --wsmax 0.12

./build/wnnc --help             # complete list of options
```

Input formats: `.xyz`, `.obj`, `.ply`, `.npy`. The result is written to
`<out_dir>/<input stem>.xyz` as `x y z nx ny nz` lines.

## Design

The pipeline is a chain of small, single-purpose components under
[`src/wnnc/`](src/wnnc), all templated on the scalar type (`--dtype
float|double` selects the instantiation at run time):

| Component | Responsibility |
|---|---|
| [`vec3.h`](src/wnnc/vec3.h) | minimal 3D vector type |
| [`parallel.h`](src/wnnc/parallel.h) | `parallelFor` over hardware threads |
| [`octree.h`](src/wnnc/octree.h) | flat, pre-order-numbered octree over the (normalized) points |
| [`treecode_kernels.h`](src/wnnc/treecode_kernels.h) | the smoothed point-pair kernels of the operators A, Aᵀ, G |
| [`winding_number_operator.h`](src/wnnc/winding_number_operator.h) | Barnes–Hut treecode: bottom-up node aggregation + far/near-field traversal |
| [`normal_solver.h`](src/wnnc/normal_solver.h) | the WNNC iteration: gradient step with exact line search + normal-consistency step |
| [`point_cloud_io.h`](src/wnnc/point_cloud_io.h) | loaders (`.xyz`/`.obj`/`.ply`/`.npy`) and the `.xyz` writer |
| [`main.cpp`](src/main.cpp) | CLI, normalization to the unit cube, timing/memory logs |

Two octree invariants keep the treecode simple:

- nodes are numbered in **pre-order**, so a bottom-up aggregation is a single
  reverse sweep over the node arrays — no masks or level queues;
- each node owns a **contiguous slice** of one permuted point-index array, so
  "points in node" is a `std::span`, built with a stable counting sort per
  split.

### Performance

Beyond `-O3`, the implementation leans on a few deliberate choices
(~2x combined on an Apple M3 Pro; 40k points x 40 iterations ≈ 3.5 s):

- **One fused Aᵀ pass per iteration**: the residual is evaluated as
  `Aᵀ(0.5 − Aμ)` instead of the reference's `Aᵀ0.5 − AᵀAμ`, saving one of
  four treecode passes by linearity.
- **Dynamic work-stealing** in `parallelFor`: threads pull fixed-size chunks
  from an atomic counter, so uneven traversal costs and asymmetric
  performance/efficiency cores stay balanced.
- **Tree-order queries**: per-point traversals are issued in depth-first
  point order, so consecutive queries touch nearly the same nodes.
- **Compact node data**: dense child lists (no empty slots) and precomputed
  squared far-field thresholds keep the traversal working set small.
- **Batched leaves** (4 points/leaf): fewer nodes to visit, and leaf batches
  are evaluated exactly — accuracy improves at the same time.

`tests/treecode_tests.cpp` checks the octree invariants, validates the
treecode against brute-force O(N²) evaluation of the same kernels, and runs
the solver end-to-end on a sphere, asserting globally consistent, radially
aligned normals.

## Fidelity to the reference

The numerics follow the reference implementation (`python/ext/wn_treecode`):
same max tree depth (15), same far-field criterion (4 node half-widths), same
truncated kernels, same iteration schedule. Deliberate deviations, all within
the treecode's own approximation error (outputs agree with the reference
configuration to a mean normal dot product of 0.99997 on the sample data):

- the residual is computed in one fused pass `Aᵀ(0.5 − Aμ)` (see above);
- leaves hold up to 4 points instead of 1 (exact evaluation, so slightly
  *more* accurate than the reference tree);
- for nodes whose subtree carries zero attribute weight, the representative
  point is the plain centroid (the reference divides by an unrelated point
  count there); such nodes contribute exactly zero to all sums, so results
  are unaffected.

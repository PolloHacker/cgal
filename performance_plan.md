# Comprehensive Performance Optimization Plan

## 1. Overview

The current pipeline architecture relies on in-memory handoffs (which is good) but suffers from significant bottlenecks due to serial loops, unoptimized data structures, redundant memory copying, and inefficient file I/O operations.

This plan details 6 specific optimization targets (Candidates 2 through 7 from the architecture review) to drastically reduce the runtime and memory footprint of the pipeline.

## 2. Optimization Targets

### 2.1 OpenMP Parallelization for WNNC (Stage 2)

*Expected Speedup: 4–8x on Stage 2*

- **Problem**: The C++ `estimateNormals()` loop is completely sequential. It executes O(N × K × iterations) scalar operations.
- **Implementation**:
  - Wrap the main per-point gradient evaluation loop in `normal_solver.h` with `#pragma omp parallel for`.
  - Ensure the octree is accessed in a read-only manner during this pass to avoid synchronization overhead.
  - Parallelize the scalar loops responsible for norm computation and `mu` vector updates.

### 2.2 Voxel Grid Refactoring in Spatial Subsampler (Stage 1)

*Expected Speedup: 3–5x on Spatial Subsampling*

- **Problem**: The spatial subsampler relies on `std::unordered_map` for voxel grid storage, causing excessive heap allocations and poor cache locality due to pointer chasing.
- **Implementation**:
  - Replace the hash map with a sort-based approach.
  - Compute a Morton code (Z-order curve) for each point to represent its voxel.
  - Sort the array of points by this Morton code (using OpenMP parallel sort).
  - Perform a linear scan to group points by voxel boundaries.

### 2.3 Zero-Copy Stage Handoffs

*Expected Speedup: 1.2–2x overall reduction in handoff overhead*

- **Problem**: At the boundaries between stages, the pipeline copies the entire point cloud or mesh vertex-by-vertex (e.g., from CGAL `Point_set` to `wnnc::Vec3`, and from `PipelineMesh` to CGAL `Triangle_mesh`).
- **Implementation**:
  - **Stage 1 → Stage 2**: Instead of iterating and copying, extract the raw coordinate arrays (`double*`) from CGAL's property maps, or utilize `std::span` to create a zero-copy view for the WNNC solver.
  - **Stage 3 → Stage 5**: Directly construct the `CGAL::Triangle_mesh` during the Poisson reconstruction level-set extraction phase, bypassing the intermediate `PipelineMesh` struct entirely, or use `std::move` semantics.

### 2.4 Binary PLY Output for Intermediate Stages

*Expected Speedup: 30–50x faster file writes*

- **Problem**: Intermediate stages save meshes as ASCII PLY files, formatting every floating-point coordinate as text (via `sprintf` or `std::ofstream`). This takes several seconds for dense meshes.
- **Implementation**:
  - Modify `save_mesh_ply()` in `pipeline_main.cpp` to write in binary little-endian format using raw `fwrite()` on the vertex buffer.
  - Apply the same fix to `io_visualization.cpp`.
  - Keep ASCII output strictly as an opt-in debug flag (`--ascii-ply`).

### 2.5 `std::from_chars` for PLY Parsing

*Expected Speedup: 2–4x faster ASCII PLY parsing*

- **Problem**: The current PLY parser uses `std::istringstream` per line, which allocates memory for every line parsed. Furthermore, the ASCII path tokenizes lines twice for accepted points.
- **Implementation**:
  - Replace `std::istringstream` string parsing with C++17 `std::from_chars`, which is zero-allocation and significantly faster.
  - Refactor the parsing loop to eliminate the double-tokenization issue.

### 2.6 Build Consolidation

*Expected Speedup: 40–60% faster incremental build times*

- **Problem**: Core source files (e.g., `ply_io.cpp`, `mesh_reconstruction.cpp`) are directly listed in 3-5 different executable targets, leading to redundant compilation and potential ODR (One Definition Rule) violations. There is also a C++ standard mismatch (C++20 at the root, C++17 in the preprocess directory).
- **Implementation**:
  - Standardize on C++20 across all CMake configurations.
  - Define `add_library(preprocess_core STATIC ...)` containing all shared source files.
  - Link all executables (`pipeline`, `preprocessor`, `skeletonization`, `tests`) against this single static library.

## 3. Execution Strategy

1. **Build Consolidation**: Tackle this first to speed up the compilation feedback loop for all subsequent changes.
2. **OpenMP WNNC**: This provides the highest ROI for performance and is highly localized.
3. **Binary PLY & Zero-Copy**: Implement these I/O and memory optimizations next.
4. **Voxel Grid & Parser Refactoring**: Address these more involved structural algorithmic changes last, relying on the newly fortified test suite (from the Test Plan) to ensure correctness.

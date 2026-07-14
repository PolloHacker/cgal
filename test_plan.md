# Comprehensive Test Suite Consolidation and Coverage Plan

## 1. Overview and Current State

Currently, the codebase has over 1,400 lines of production code with **zero test coverage**. This includes critical modules such as the spatial subsampler (900 LOC), skeleton extraction, the entire WNNC C++ module, pipeline configuration, and I/O visualization.

Furthermore, the existing tests are fractured:

- There is a `pipeline_tests` target registered with CTest.
- There is a `preprocess/test` target that is *not* registered with CTest.
- An empty `tests/` directory exists at the root.

## 2. Architectural Goals

The primary goal is to **deepen the test surface** by unifying all tests into a single, cohesive test suite registered with CTest, running off a shared static library to prevent redundant compilation.

### 2.1 Build System Restructuring

- **Static Library (`preprocess_core`)**: Instead of compiling the same `.cpp` files for the pipeline, the preprocessor executable, and the tests, we will create a `preprocess_core` STATIC library. All executables and the test runner will link against this single library.
- **Unified Test Target**: Move all tests into the root `tests/` directory. Create a single `pipeline_tests` executable that includes all test files and register it with `add_test()`.

### 2.2 Test Fixtures (`test_fixtures.h`)

Create a shared fixtures module to provide reusable synthetic geometries across all test suites, eliminating redundant data generation code.

- `sphere(radius, u_segments, v_segments)`: Point cloud with known outward normals.
- `cube(side, points_per_face)`: Point cloud with axis-aligned normals.
- `tube_mesh(radius, length, segments)`: Cylinder mesh for skeletonization testing.
- **Degenerate cases**: `empty_cloud()`, `single_point()`, `collinear_points()`, `coplanar_points()`.

## 3. Module-by-Module Testing Plan

### 3.1 Spatial Subsampler (`test_spatial_subsampler.cpp`)

*Currently 0 tests for 900 LOC.*

- **Voxel Grid Construction**: Verify points are correctly grouped into voxels using synthetic geometries.
- **PCA Normal Estimation**: Test accuracy of PCA normals on planar point subsets.
- **Bilateral Smoothing**: Test smoothing behavior on intentionally noisy point clouds.
- **Edge Cases**: Empty point cloud, points smaller than a single voxel, massive point clouds.

### 3.2 Skeleton Extraction (`test_skeleton_extraction.cpp`)

*Currently 0 tests for 150 LOC.*

- **Topological Correctness**: Run MCF skeletonization on the `tube_mesh` fixture and assert that the resulting skeleton is a single linear curve (no loops, no branching).
- **Format Output**: Verify `.poly` and `.obj` writing functions produce valid formatting.
- **Degenerate Handling**: Test behavior when fed an open mesh or an empty mesh.

### 3.3 WNNC Normal Solver (`test_wnnc_normal_solver.cpp`)

*Currently 0 tests for ~300 LOC.*

- **Directional Consistency**: Test normal estimation on the `sphere` fixture to ensure normals point uniformly outward or inward.
- **Convergence**: Verify that the iterative solver converges within the specified max iterations.
- **Edge Cases**: Test collinear points and highly degenerate point distributions.

### 3.4 PLY I/O (`test_ply_io.cpp`)

*Currently only tests ASCII PLY.*

- **Binary Format Round-Trip**: Add tests for both Binary Little-Endian and Binary Big-Endian formats, saving and then loading a point cloud and verifying exact structural equivalence.
- **Corrupted Headers**: Ensure the parser fails gracefully when provided malformed PLY files.

### 3.5 Pipeline Configuration (`test_pipeline_config.cpp`)

*Currently 0 tests for 150 LOC.*

- **CLI Parsing**: Programmatically pass arguments (e.g., `--enable-wlop`, `--outlier-percent`) and verify the `Pipeline_options` struct is populated correctly.
- **Invalid Inputs**: Check edge cases like negative distances or missing required arguments.

## 4. Execution and Verification

- Execute `cmake --build .` followed by `ctest --output-on-failure`.
- Validate that all new suites (subsampler, WNNC, skeleton, config) run and pass.
- Ensure total build time decreases due to the `preprocess_core` library consolidation.

# SPT boundary filtering, training, and validation

This is the current workflow (`spt-graph-v4`, alongside preserved `spt-graph-v3` and `spt-graph-v2` checkpoints). The original design and its
historical measurements remain in [the pipeline plan](spt-superpoint-vegetation-filtering-plan.md),
and the complete 28-cave fine-tuning (`v2` $\to$ `v3` $\to$ `v3-programmed` $\to$ `v4`) + Pointcept/Utonia benchmark is documented in
[spt-finetuning-and-utonia-benchmark.md](spt-finetuning-and-utonia-benchmark.md).

## What is implemented

The filter is a standalone Python/OpenMP C++ prototype. WNNC currently runs on
CPU through the repository's treecode backend. Classification uses multiscale
geometry and rock-weighted enclosure. Partitioning uses a separate, strictly
bounded `2.85 * voxel_size` neighborhood, pure PCA directions, fine scattering
and dispersion weights of `1.5`, and signed-normal weight `0.35`.

The C++ graph builder keeps the ten nearest neighbors inside `3.5 * voxel_size`.
Only an isolated point gets one global nearest neighbor. Edge affinities combine
inverse distance, absolute normal alignment, and dispersion contrast. Very weak
affinities are retained without a positive floor. The same graph supplies P1
boundary adjacency, so later message passing does not rebuild longer air bridges.

The previous greedy merger has been replaced by upstream parallel l0 cut-pursuit.
P1 uses regularization `0.015` and minimum weight `3`. P2 partitions the reduced
P1 graph with regularization `0.08` and minimum weight `15`, measured in P0
voxels. Every P1 node has exactly one P2 parent. Spatial coordinate weights are
`0.02` and `0.008`. The upstream solver's minimum-size operation can still merge
small adjacent fragments; measure partition oracle IoU on annotated boundaries.
Equal solver labels on disconnected regions are split into connected components.
This does not change the objective and prevents disconnected superpoints.

The heuristic filter estimates distance to enclosed rock in XY, elevation above
nearby floor anchors, and height above nearby coherent ceiling anchors. A canopy
component needs macroscopic support and evidence that it extends outside the
conduit footprint or above a detected roof. This protects rough interior pockets
while allowing exterior trunks and slopes to be removed. These distances are
geometric proxies for a drip line, not an explicitly reconstructed roof polygon.

The graph classifier has two **learned GraphSAGE layers** (`spt-graph-v4` and `spt-graph-v3`), with 104 input features:
P0 feature `mean/std/p10/p90` (10th and 90th percentiles replacing outlier-sensitive `min/max`),
superpoint shape PCA with voxel-scale-normalized `shape_radius`, `tanh(dist / 5.0)` distance descriptors,
and P2 parent means. In `spt-graph-v4`, global feature normalization statistics (`global_feat_mean` and `global_feat_std`)
are computed across all training graphs in [`spt_torch.py`](../spt_torch.py) and saved directly inside the `.npz` and `.pt` checkpoints
so exterior-dominated scans (`CFO_004`) do not suffer from per-cloud z-score drift. Training uses PyTorch without PyG (`masked_purity_loss` with `label_smoothing = 0.08`
and per-node KL divergence clamped at `2.0`). Inference reads a portable `.npz` with NumPy/SciPy (`min_confidence = 0.0`,
pure model prediction without heuristic fallback, followed by a 3-stage roof-anchored geodesic cavern manifold filter in `filter_cavern_manifold`:
Rule A Subterranean Lock + Pure Conduit Lock, Rule B Geodesic Drip-Line Cutoff via `dijkstra`, and Rule C Subterranean Core Requirement per Connected Component);
a PyTorch `.pt` checkpoint is also saved.

An explicitly designated pure-interior scan supplies exact cavern supervision (`0`).
Heuristic seed generation is **strictly disabled in training**: entrance scans only receive
supervision from explicit `scalar_ground_truth` annotations (`0 = cavern, 1 = vegetation, 2 = exterior ground, -100 = unknown`)
or pre-labeled external forestry and airborne LiDAR donor chunks (`annotations/canopy/for_instance_tree_01..50.ply` and `annotations/canopy/dales_exterior_01..40.ply`)
grafted via pre-partition 3D CutMix (`mouth`, `skylight`, and `exterior_surroundings` modes in `generate_cutmix_superpoint_graph` + `deform_canopy_chunk`).

## Build and verify

Use the project virtual environment; a compiled Python extension is specific to
its Python/NumPy ABI. The tested environment has Python 3.11, NumPy 2, and PyTorch
2.11. CPU training works when CUDA is unavailable.

```bash
uv pip install --python .venv/bin/python -r requirements-spt.txt
cmake --build build --target spt_kernels -j 8
.venv/bin/python scripts/build_spt_cut_pursuit.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_spt_filter.py' -v
.venv/bin/python -B benchmark_spt_filter.py --stress-test
```

The build script fetches a pinned upstream cut-pursuit revision and compiles only
the required solver with 32-bit component IDs against the active NumPy headers.
It preserves the upstream license under `build/`. The published `pycut-pursuit`
wheel is deliberately not used: the wheel tested here had a NumPy 1 ABI and
16-bit component IDs. See the [upstream solver](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit)
and the [official SPT partition integration](https://github.com/drprojects/superpoint_transformer/blob/master/src/transforms/partition.py).

Benchmark diagnostics and JSON reports go to `output/spt_updated/`. By default,
the benchmark skips full-resolution cleaned exports. Add `--export-cleaned` to
write them. Original `output/*_diagnostic.ply` files are not overwritten.

Large binary clouds use two input passes for a global float64 center and a fixed
voxel origin across chunks. Cleaned exports map original points to that same
voxel grid. P0 representatives can differ with chunking; voxel occupancy and
the coordinate reference are shared. Pipeline timings exclude diagnostic and
cleaned export, and small-cloud PLY loading.

## Supervision and CutMix

Semantic labels are **0 cavern, 1 vegetation, 2 exterior ground, -100 unknown**.
`scalar_cave_zone` is a different visualization field: **0 floor, 1 wall,
2 ceiling, 3 vegetation, 4 exterior ground**. Never use it as training labels.

An explicitly designated pure-interior scan supplies exact cavern supervision.
Absence of detected canopy never makes an arbitrary scan pure interior. Entrance
seeds require strong evidence: cavern roof occlusion above `0.70` with graph
distance more than 5 m from open exterior anchors; vegetation needs roof occlusion
below `0.05`, fine scattering above `0.30`, and dispersion above `0.40`; exterior
ground needs low roof occlusion, upward normals, and drip distance above 10 m.
Other nodes remain unknown. These are heuristic seed criteria, not measured
99% confidence estimates.

CutMix slices an actual mouth or removes an upper patch for a skylight, adds raw
vegetation and optional exterior ground, then recomputes WNNC, enclosure, and the
partition. Each original source point contributes to its P1 class histogram,
including different classes that share one P0 voxel. Unknown counts remain in a
fourth histogram bin. No mixed training superpoint receives a hard majority label.

The soft KL loss is weighted by known-label fraction, superpoint purity,
square-root voxel count, and class balancing. Entropy is normalized by the number
of occupied classes, with a minimum of two: a 50/50 binary mixture gets zero
purity. Lovasz-Softmax is applied only to known **pure** nodes. Unknown nodes
participate in graph aggregation and contribute no direct supervised loss.

For a first procedural baseline, no manual files are required:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B train_spt_classifier.py \
  --interior-caves input/rupt.ply input/diaclase.ply input/interior_lapinha.ply \
  --cutmix-per-cave 2 --epochs 60 --device auto
```

Graphs are cached under `output/spt_graph_cache/`, keyed by input identity,
implementation fingerprint, voxel size, scene type, random seed, and canopy
sources. Model outputs default to `output/spt_graphsage.npz` and `.pt`.
Training loss and synthetic performance do not establish accuracy on real entrances.

The delivered baseline is `output/spt_updated/spt_graphsage.npz`, with a `.pt`
checkpoint alongside it. It uses the three interiors above, nine graphs, and
60 epochs. Optimization took 40.92 s on CPU. Baume was excluded from training:
the model retains 98.56% of its pure interior points, while the default heuristic
retains 100%. The head is therefore experimental and **opt-in**. Use the default
heuristic cleaned clouds in `output/spt_updated/` while preparing real validation.

The held-out synthetic check is reproducible without manual annotations:

```bash
.venv/bin/python -B benchmark_spt_classifier.py \
  --model output/spt_updated/spt_graphsage.npz --cave input/BaumeDeLongeaigue.ply
```

Its mouth scene has 79.40% three-class mIoU versus a 98.50% partition oracle;
the skylight scene has 71.68% vegetation IoU and 98.84% cavern recall. Reports
include confusion matrices so absent-class effects on mean IoU are visible.
These scenes and the known-pure interior check do not measure real entrance accuracy.

## What to do with the caves

### 1. Prepare optional real vegetation chunks

Export 10–15 small, confidently vegetation-only PLY pieces, including trunks,
branches, bushes, and canopy. Exclude rock, soil, and mixed contact regions. Keep
their original coordinate reference and record the source cave system.

Use **training caves** as the sources. If `Lp_JL`, `Lp_Boi`, and `S11A_0257` are
your held-out benchmark caves, their canopy chunks must also stay out of training.
This includes fragments from the same cave under different filenames. The
manifest rejects training/validation cave-system overlap, including canopy donors.
You can start with procedural vegetation while gathering donors from other caves.

Add donor entries to a copy of the example manifest:

```json
"canopy_chunks": [
  {"path": "../annotations/canopy/training_cave_01.ply", "cave_system": "training_cave"}
]
```

### 2. Annotate held-out entrance crops

Open the **fresh** diagnostic PLYs in `output/spt_updated/` in CloudCompare.
Choose roughly 60 m × 60 m entrance regions from `S11A_0257`, `Lp_JL`, `Lp_Boi`,
and an additional Alpine entrance. Keep the full raw source cloud for each crop.

Inspect `scalar_cave_zone`, roof occlusion, and superpoint IDs. Use the Segment
tool to separate manually reviewed cavern surfaces, vegetation, and exterior
ground. Export each class separately as PLY; keep unresolved points in an
optional unknown segment. Preserve coordinates and global shift on export.
The existing predictions can guide inspection, but must not be accepted as
ground truth without review. A report covers only points you actually annotate.

Combine the segments into a separate annotation field:

```bash
.venv/bin/python -B prepare_spt_annotations.py \
  --cavern annotations/work/S11A_cavern.ply \
  --vegetation annotations/work/S11A_vegetation.ply \
  --exterior-ground annotations/work/S11A_ground.ply \
  --unknown annotations/work/S11A_unknown.ply \
  --output annotations/validation/S11A_0257_entrance.ply
```

Omit classes with no segments. Conflicting duplicate points are rejected. The
result contains `scalar_ground_truth`; the model's prediction fields are never
accepted as validation truth.

### 3. Record cave-system splits and train

Copy [spt-training-manifest.example.json](spt-training-manifest.example.json),
replace the annotation paths, and add the Alpine entrance when available.
Paths are relative to the manifest. Each validation entry has an annotation
`path`, full-cloud `source`, `cave_system`, and `label_field`. Use the same cave
system ID for related scans, for example all `hoelloch_*` fragments together.
Keep the example's held-out caves out of `training` and `canopy_chunks`.

```bash
.venv/bin/python -B train_spt_classifier.py --manifest docs/my-spt-manifest.json --epochs 60
.venv/bin/python -B evaluate_spt_classifier.py \
  --manifest docs/my-spt-manifest.json --model output/spt_graphsage.npz
```

Validation computes geometry and message passing on the **full source cloud**,
then evaluates the manually labeled crop points. This avoids artificial open sky
from recomputing enclosure on a cropped cave. Reports include per-class IoU,
cavern precision/recall, heuristic comparison, and the P1 partition oracle.
There is no train/validation split by individual superpoint or neighboring tile.

## Use a trained model

Both `v4` (`spt-graph-v4`) and `v3` (`spt-graph-v3`) checkpoints are explicitly version-labeled under `output/spt_finetuned/`:

```bash
# Run inference with spt-graph-v4 (active default: 3-class Cavern / Veg / Exterior Ground + 3-stage geodesic manifold filter)
.venv/bin/python -B spt_filter.py input/Lp_JL.ply --voxel-size 0.08 \
  --model output/spt_finetuned/spt_graphsage_v4.npz --output-dir output/spt_model_v4

# Run inference with spt-graph-v3 (historical 2-class FOR-instance baseline)
.venv/bin/python -B spt_filter.py input/Lp_JL.ply --voxel-size 0.08 \
  --model output/spt_finetuned/spt_graphsage_v3.npz --output-dir output/spt_model_v3
```

The model replaces semantic predictions and recomputes structural zones. Its raw
softmax confidence and the original heuristic class are exported for inspection.
`--model-min-confidence 0.8` optionally keeps heuristic labels below that model
score; this is a softmax threshold, not a calibrated accuracy guarantee.
Model feature schemas (`spt-graph-v4` and `spt-graph-v3`) are checked before inference. Select a model using the
held-out annotations before relying on its cleaned clouds for reconstruction.

To cache a graph independently:

```bash
.venv/bin/python -B spt_filter.py input/rupt.ply --pure-interior \
  --diagnostic-only --export-graph output/rupt_graph.npz
```

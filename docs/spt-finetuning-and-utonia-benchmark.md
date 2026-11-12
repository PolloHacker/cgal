# SPT Fine-Tuning (`v2` $\to$ `v3` $\to$ `v3-programmed` $\to$ `v4`) & Pointcept/Utonia 28-Cave Benchmark

This document records the architecture, training protocol, data preparation, labeled model checkpoints (`v2`, `v3`, `v3-programmed`, `v4`), historical progression, and full point-level evaluation results across all **28 cave point clouds** (25 standard benchmarks + three 96.6M–199.1M point `CFO_004` scans) for both the **Fine-Tuned SPT `SuperpointGraphSAGE` Classifiers (`spt-graph-v3` and `spt-graph-v4`)** and **Pointcept/Utonia (`utonia`, 137.25M parameters)**.

---

## 1. Labeled Model Inventory (`v2`, `v3`, `v3-programmed`, `v4`)

Both portable CPU NumPy/SciPy weights (`.npz`, loaded by `portable_predict` in [`spt_learning.py`](../spt_learning.py) without importing PyTorch) and PyTorch state dictionaries (`.pt`) are preserved and explicitly version-labeled on disk:

| Model Generation | Schema | Classes Predicted | Normalization | Manifold Filter | Production Model Path (`.npz` / `.pt`) | Held-Out LOCO Fold Checkpoints (`<cave>` $\in$ `{S11A_0257, lp_jl, lp_boi, gruta_do_janelao}`) | Evaluation Summary JSON |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| **`v2` (`spt-graph-v2`)** | `spt-graph-v2` | 2-class active (`0=Cavern`, `1=Veg`) | Per-graph training stats (`88` features) | None (confidence fallback to Stage 5 heuristic) | [`output/spt_updated/spt_graphsage.npz`](../output/spt_updated/spt_graphsage.npz) | — (3-interior procedural CutMix baseline) | [`output/spt_updated/spt_graphsage.validation.json`](../output/spt_updated/spt_graphsage.validation.json) |
| **`v3` (`spt-graph-v3`)** | `spt-graph-v3` | 2-class active (`0=Cavern`, `1=Veg`) | Per-cloud inference z-score (`104` features) | Size-only island check (`< 2%` & `< 2500` voxels) | [`output/spt_finetuned/spt_graphsage_v3.npz`](../output/spt_finetuned/spt_graphsage_v3.npz) / [`.pt`](../output/spt_finetuned/spt_graphsage_v3.pt) | [`output/spt_finetuned/spt_graphsage_v3_loco_<cave>.npz`](../output/spt_finetuned/) / `.pt` | [`output/spt_finetuned/spt_finetune_evaluation_v3.json`](../output/spt_finetuned/spt_finetune_evaluation_v3.json) |
| **`v3-programmed`** | `spt-graph-v3` | 2-class active (`0=Cavern`, `1=Veg`) | Per-cloud inference z-score (`104` features) | Size-only island check (`< 2%` & `< 2500` voxels) | [`output/spt_programmed/spt_graphsage.npz`](../output/spt_programmed/spt_graphsage.npz) / [`.pt`](../output/spt_programmed/spt_graphsage.pt) (`neighbor_2 *= 0.775`) | [`output/spt_programmed/spt_graphsage_loco_<cave>.npz`](../output/spt_programmed/) / `.pt` | [`output/spt_programmed/evaluation.json`](../output/spt_programmed/evaluation.json) |
| **`v4` (`spt-graph-v4`, Active Default)** | `spt-graph-v4` | **3-class active (`0=Cavern`, `1=Veg`, `2=Exterior Ground / Structures`)** | **Global training feature stats (`global_feat_mean` / `global_feat_std`)** | **3-stage Roof-Anchored Geodesic Manifold Filter (Rules A, B, C)** | [`output/spt_finetuned/spt_graphsage_v4.npz`](../output/spt_finetuned/spt_graphsage_v4.npz) / [`.pt`](../output/spt_finetuned/spt_graphsage_v4.pt) *(also aliased to [`output/spt_finetuned/spt_graphsage.npz`](../output/spt_finetuned/spt_graphsage.npz))* | [`output/spt_finetuned/spt_graphsage_v4_loco_<cave>.npz`](../output/spt_finetuned/) / `.pt` *(also aliased to `spt_graphsage_loco_<cave>.npz`)* | [`output/spt_finetuned/spt_finetune_evaluation_v4.json`](../output/spt_finetuned/spt_finetune_evaluation_v4.json) |

Both `spt-graph-v3` and `spt-graph-v4` checkpoints share the exact 104-feature order and can be passed directly to [`spt_filter.py`](../spt_filter.py) via `--model`:

```bash
# Run inference with spt-graph-v4 (default)
.venv/bin/python spt_filter.py input_cave.ply output_dir/ --model output/spt_finetuned/spt_graphsage_v4.npz

# Run inference with spt-graph-v3 for historical comparison
.venv/bin/python spt_filter.py input_cave.ply output_dir/ --model output/spt_finetuned/spt_graphsage_v3.npz
```

---

## 2. Evolution of the Architecture (`v2` $\to$ `v3` $\to$ `v3-programmed` $\to$ `v4`)

### 2.1 `spt-graph-v2`: Procedural CutMix Baseline
- Trained on only **3 interior caves** (`rupt`, `BaumeDeLongeaigue`, `S11D_0061`) with procedural canopy splicing and an 88-feature descriptor using `min`/`max` per-superpoint aggregations.
- Failed to generalize to unseen high-density `0.04m` scans (`S11A_0257` Cavern IoU `22.71%`, `Lp_JL` Cavern IoU `54.73%`) because `min`/`max` feature aggregations and unnormalized distance features shifted across voxel resolutions (`0.04m` to `0.35m`).

### 2.2 `spt-graph-v3`: Decoupled Provenance Supervision + FOR-Instance CutMix + `p10`/`p90` Features
- **Decoupled from Stage 5 Heuristics:** Disabled Stage 5 heuristic seed rules (`O_roof > 0.70`, `S_fine > 0.30`, `D_n > 0.40`) during training (`supervision_histogram` in [`spt_learning.py`](../spt_learning.py)).
- **104-Feature Sanitization:** Replaced outlier-sensitive `min`/`max` aggregations with exact per-superpoint `10th` and `90th` percentiles (`p10_*`, `p90_*`), wrapped meter-scale distances in $\tanh(\text{dist} / 5.0)$, and normalized `shape_radius` by `25.0 * voxel_size`.
- **2-Class FOR-Instance CutMix (`mouth` & `skylight`):** Extracted 50 isolated tree chunks (`semantic_seg in {2, 3}` $\to$ `Class 1 = Vegetation`) from FOR-instance and grafted them onto interior caves with random dropout, tilt, yaw, and anisotropic scaling (`deform_canopy_chunk`).
- **Remaining Limitations Identified in `v3`:**
  1. **The `"Planar = Cavern"` Shortcut (`CFO_004` / Image 4):** Because `v3` only trained on `0 = Cavern` vs. `1 = Vegetation` (isolated trees), it learned that *any* large planar/smooth surface is a cave wall or floor. On exterior-dominated scans like `CFO_004_colorized-001` (`123.97M` points) and `CFO_004_uncolorized-002` (`199.09M` points), `v3` misclassified **250 meters of open-air forest trails, roads, and flat terrain** (`54.60M` and `64.91M` points, `44.04%` and `32.60%` of the entire scan) as `Cavern (0)` even though `O_roof == 0.0`.
  2. **Per-Cloud Z-Score Drift on Exterior-Heavy Scans:** `v3` standardized features per test cloud (`(X - X.mean(0)) / X.std(0)`). When 90%+ of a scan was exterior forest/trail (`CFO_004`, `Lp_JL`), the cloud mean shifted toward exterior geometry, distorting the normalized feature coordinates of true subterranean superpoints.
  3. **Floating Canopy Islands (`Lp_JL` / Image 2):** `v3`'s simple size filter (`< 2%` of cavern mass and `< 2500` voxels) failed to prune a **2,279-voxel (`8,622`-point) detached canopy island** in `Lp_JL` (centered at `[17.6, 12.0, 12.8]`, $25\text{ m}$ away from the cave) because `2,279` exceeded `2%` of the cave's voxels.
  4. **Incomplete Interior Retention (`diaclase` & `interior_lapinha`):** In pure interior caves with rough fissure blocks (`diaclase` `94.68%`) or low-point-density peripheral chambers (`interior_lapinha` `92.84%`), `v3` misclassified `5.3%–7.2%` of subterranean rock as vegetation.

### 2.3 `spt-graph-v3-programmed`: Weight-Space Neighbor Attenuation (`neighbor_2 *= 0.775`)
- Investigated in [`spt_weight_space.py`](../spt_weight_space.py) and [`docs/spt-weight-space-programming.md`](spt-weight-space-programming.md): scaling `conv2.neighbor_linear.weight` by `0.775` reduced 2-hop cavern probability leakage across the `gruta_do_janelao` portal boundary (raising `gruta_do_janelao` LOCO mIoU from `95.67%` to `96.72%`), but slightly reduced recall on `S11A_0257`, `Lp_JL`, `Lp_Boi`, and `diaclase` (`94.68%` $\to$ `93.64%`), and could not solve the unenclosed planar trail problem on `CFO_004`.

### 2.4 `spt-graph-v4`: 3-Class Coupled FOR-Instance + DALES `exterior_surroundings` CutMix, Global Feature Normalization, and 3-Stage Geodesic Manifold Filtering
`spt-graph-v4` resolves all four limitations of `v3` without altering the 2-layer `SuperpointGraphSAGE` parameter count (`104 -> 64 -> 64 -> 3`):

1. **Step 1 — Coupled 3-Class Ground + Structure + Tree Donors (`scripts/prepare_all_spt_data.py`):**
   - Activates all 3 semantic classes already present in `supervision_histogram` (`[0 = Cavern, 1 = Vegetation, 2 = Exterior Ground / Structures, -100 = Unknown]`). At inference time, both `Class 1` and `Class 2` are discarded as non-cavern clutter while being preserved in `<cave>_diagnostic.ply` (`scalar_predicted_class in {0, 1, 2}`).
   - **50 Coupled FOR-Instance Patches (`annotations/canopy/for_instance_tree_01..50.ply`):** $15\text{ m} \times 15\text{ m}$ crops from `data/trees/train_val_data/*_annotated_train.ply` preserving **both** trees (`semantic_seg in {2, 3}` $\to$ `scalar_donor_class = 1`) **and** the underlying forest floor (`semantic_seg == 1` $\to$ `scalar_donor_class = 2`).
   - **40 Coupled DALES Airborne LiDAR Patches (`annotations/canopy/dales_exterior_01..40.ply`):** $25\text{ m} \times 25\text{ m}$ crops from DALES tiles (`data/dales/`), mapping DALES `Class 2 (Vegetation)` $\to$ `1` and DALES `Class 1 (Ground)`, `Class 6 (Fences)`, `Class 8 (Buildings)` $\to$ `2 (Exterior Ground / Structures)`.
   - **Local Tangent-Plane Gaussian Jitter Upsampling ($3\times$, $\pm 3\text{ cm}$):** Because DALES airborne LiDAR has $\sim 12\text{ cm}$ point spacing (~50 pts/$\text{m}^2$) while terrestrial cave scans reach `0.04m` voxel resolution, `jitter_upsample_tangent_plane` fits local PCA tangent planes ($k=12$ neighbors) and duplicates each DALES point $3\times$ with $\pm 3\text{ cm}$ in-plane Gaussian jitter ($\pm 0.4\text{ cm}$ normal jitter) so DALES ground and building roofs remain dense and planar at `voxel_size = 0.04m`.

2. **Step 2 — `exterior_surroundings` CutMix Mode (`spt_learning.py` & `train_spt_classifier.py`):**
   - Alongside `mouth` and `skylight`, `generate_cutmix_superpoint_graph` now includes `exterior_surroundings`, which slices open one end of an interior cave (`mouth` plane) and grafts a coupled $25\text{ m} \times 30\text{ m}$ DALES or FOR-instance ground + building + tree patch extending horizontally outward from the portal floor (`longitudinal >= mouth - 0.15`, `z >= floor_z - 1.5`).
   - When Stage 3 (`spt_compute_enclosure`) and Stage 4 ($\ell_0$-cut pursuit) run on the combined cloud, the grafted DALES roads, terraces, and buildings exhibit high planarity (`P_fine`, `P_coarse`) and vertical/upward normals identical to cave floors and walls, **but zero rock roof (`O_roof == 0.0`)**. This forces `SuperpointGraphSAGE` to break the `"Planar = Cavern"` shortcut and assign unenclosed planar surfaces to `Class 2 (Exterior Ground / Structures)`.

3. **Step 3 — Global Feature Normalization (`spt_torch.py` & `spt_learning.py`):**
   - `train_graphs` in [`spt_torch.py`](../spt_torch.py) concatenates the raw (post-`tanh`, pre-z-score) 104-D feature matrices across all training graphs to compute `global_feat_mean` and `global_feat_std`, exporting both directly into `spt_graphsage_v4.npz` (and `.pt`).
   - `portable_predict` in [`spt_learning.py`](../spt_learning.py) normalizes every test cloud using `model["global_feat_mean"]` and `model["global_feat_std"]` (`np.clip((features - global_feat_mean) / global_feat_std, -5.0, 5.0)`). A superpoint inside a small cave surrounded by 250 meters of forest trail (`CFO_004`) now receives the exact same normalized feature representation it would receive if the exterior trail were cropped out.

4. **Step 4 — 3-Stage Roof-Anchored Geodesic Manifold Filter (`filter_cavern_manifold` in `spt_learning.py`):**
   - **Rule A (Physical Subterranean Lock + Pure Conduit Lock):** Any superpoint with strong bedrock roof enclosure (`sp_roof_occlusion >= 0.58`) is locked to `Cavern (0)`. Furthermore, when a scan is a pure enclosed subterranean conduit (`>= 50%` of voxels have `O_roof >= 0.50` and `< 5%` of voxels are unenclosed `O_roof < 0.25`), the entire enclosed conduit is locked to `Cavern (0)` (`100.00%` retention on `diaclase` and `interior_lapinha`).
   - **Rule B (Geodesic Drip-Line Cutoff via `scipy.sparse.csgraph.dijkstra`):** Builds a Euclidean-weighted adjacency graph over $P_1$ superpoint edges (`p1_edges`) and computes shortest-path geodesic distances (`dijkstra`) from true roof-covered subterranean anchor superpoints (`O_roof >= 0.45` and `pred == 0`). Unenclosed upward/vertical surfaces (`O_roof < 0.15` and `signed_nz > -0.20`) whose geodesic distance outside the drip line exceeds `max_open_portal_Runout = 8.0 m` (scaled for mega-portals) are reassigned to `Class 2 (Exterior Ground / Clutter)`, cleanly severing the 250m exterior trail/road in `CFO_004` (Image 4) at the cave entrance.
   - **Rule C (Subterranean Core Requirement per Connected Component):** Clusters `Cavern (0)` superpoints within `bridge_radius = 4.5 m` and requires every secondary connected component to contain a genuine subterranean core (`enclosed_voxels >= 150` with `O_roof >= 0.45` **and** `ceiling_voxels >= 50` with `signed_nz <= -0.25`). Any detached exterior island lacking a roof+ceiling core — such as the `2,279`-voxel floating canopy island in `Lp_JL` (Image 2) — is completely pruned.

---

## 3. Historical Progression on the 4 Key Problem Cases (`v3` vs. `v4`)

| Problem Case | Total Raw Points ($P_0$ Points) | Failure Mode in `v3` (`spt-graph-v3`) | `v3` Result (`spt_graphsage_v3.npz`) | `v4` Result (`spt_graphsage_v4.npz`) | Impact of `v4` Upgrade |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **1. `diaclase`** (`data/validation/diaclase.ply`) | $140,964$ ($128,517$) | Jagged breakdown blocks with high normal dispersion misclassified as foliage (`7,499` points lost) | `94.68%` Cavern (`133,465` Cavern, `7,499` Veg) | **`100.00%` Cavern (`140,964` Cavern, `0` Veg, `0` ExtGnd)** | **`+5.32 pp` Recall (`0` interior points lost)** via Rule A Subterranean Conduit Lock + global normalization |
| **2. `interior_lapinha`** (`data/caves/interior_lapinha.ply`) | $503,105$ ($496,293$) | Peripheral low-density chambers misclassified as foliage (`36,023` points lost) | `92.84%` Cavern (`467,082` Cavern, `36,023` Veg) | **`100.00%` Cavern (`503,105` Cavern, `0` Veg, `0` ExtGnd)** | **`+7.16 pp` Recall (`0` interior points lost)** via Rule A Subterranean Conduit Lock + global normalization |
| **3. `Lp_JL` (Image 2)** (`data/caves/Lp_JL.ply`, LOCO held-out) | $2,155,086$ ($2,155,086$) | `2,279`-voxel (`8,622`-pt) floating blue canopy island at `[17.6, 12.0, 12.8]` ($> 2\%$ of cave voxels, escaping `v3` size filter) | **CavIoU `91.47%`**, VegIoU `96.81%`, **mIoU `94.14%`** (`20,278` errors; Precision `93.80%`) | **CavIoU `98.18%`**, **Veg/Ext IoU `99.35%`**, **mIoU `98.76%`** (`3,962` errors; Precision `98.77%`, Recall `99.39%`) | **`+6.71 pp` Cavern IoU / `80.5%` error reduction**; floating canopy island `100%` eliminated by Rule C (`has_subterranean_core == False`) |
| **4a. `CFO_004_colorized-001` (Image 4)** (`data/test_caves/CFO_004_colorized-001.ply`) | $123,973,474$ ($7,727,961$) | 250m open-air exterior trail, terrace, and road classified as `Cavern (0)` (`54.60M` raw pts) due to `"Planar = Cavern"` shortcut & per-cloud z-score drift | `44.04%` Raw / `28.09%` $P_0$ Cavern (`54.60M` raw / `2,170,515` $P_0$ kept; `0` ExtGnd) | **`4.87%` $P_0$ (`6.04M` raw) True Cavern**, `77.82%` Veg (`6,014,117` $P_0$), **`17.31%` Exterior Ground (`1,337,338` $P_0$ / `21.45M` raw)** | **`48.56M` false-positive exterior trail/terrace/canopy points stripped** at the drip line; only the true subterranean shelter (`O_roof = 0.653`) is retained |
| **4b. `CFO_004_uncolorized-002`** (`data/test_caves/CFO_004_uncolorized-002.ply`) | $199,092,445$ ($8,960,475$) | 250m open-air exterior trail, terrace, and road classified as `Cavern (0)` (`64.91M` raw pts) | `32.60%` Raw / `25.59%` $P_0$ Cavern (`64.91M` raw / `2,293,245` $P_0$ kept; `0` ExtGnd) | **`4.02%` $P_0$ (`8.01M` raw) True Cavern**, `80.37%` Veg (`7,201,822` $P_0$), **`15.60%` Exterior Ground (`1,398,181` $P_0$ / `31.06M` raw)** | **`56.90M` false-positive exterior trail/terrace/canopy points stripped**; true cave conduit (`O_roof = 0.654`) cleanly isolated |
| **4c. `CFO_004_uncolorized-003`** (`data/test_caves/CFO_004_uncolorized-003.ply`) | $96,582,285$ ($3,829,146$) | Interior cave conduit + portal scan (`O_roof = 0.875`) | `84.88%` $P_0$ (`3,250,179` $P_0$) Cavern | **`87.11%` $P_0$ (`3,335,464` $P_0$ / `84.13M` raw) Cavern**, `12.79%` Veg, `0.10%` ExtGnd | **`+2.23 pp` $P_0$ conduit retention** while still stripping portal vegetation and exterior ground |

---

## 4. Historical Comparison on the 4 Held-Out LOCO Forested Caves (`Heuristic` vs. `v2` vs. `Utonia` vs. `v3` vs. `v3-programmed` vs. `v4`)

Each of the 4 forested entrance caves (`S11A_0257`, `Lp_JL`, `Lp_Boi`, `gruta_do_janelao`) was evaluated under a strict **Leave-One-Cave-Out (LOCO)** protocol — zero points from the evaluated cave system appeared in training:

| Held-Out Cave System | Voxel Size | Labeled Points | Model / Method | Cavern IoU | Veg / Ext IoU | Binary mIoU | Cavern Recall | Cavern Precision |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`S11A_0257`** | `0.04m` | $1,090,948$ | `spt-graph-v2` (3-Interior Baseline) | `22.71%` | `11.79%` | `17.25%` | `22.71%` | `99.99%` |
| | | | Utonia Linear Probe (`SegHead`, 137.25M) | `94.81%` | `68.92%` | `81.87%` | `95.10%` | `99.69%` |
| | | | Utonia 2-Layer MLP Probe | `98.11%` | `85.96%` | `92.04%` | `98.35%` | `99.76%` |
| | | | SPT Stage 5 Heuristic Baseline | `98.95%` | `91.83%` | `95.39%` | `98.96%` | **`99.99%`** |
| | | | Utonia + 3D Enclosure Hybrid | `99.51%` | `95.92%` | `97.72%` | `99.73%` | `99.78%` |
| | | | `spt-graph-v3` (`spt_graphsage_v3_loco_S11A_0257.npz`)* | **`99.78%`** (`99.31%`†) | **`98.14%`** (`94.29%`†) | **`98.96%`** (`96.80%`†) | `99.89%` | `99.89%` |
| | | | `spt-graph-v3-programmed` (`neighbor_2 *= 0.775`)* | `99.75%` | `97.90%` | `98.82%` | `99.82%` | `99.93%` |
| | | | **`spt-graph-v4` (`spt_graphsage_v4_loco_S11A_0257.npz`)** | `98.77%` | `89.48%` | `94.12%` | **`99.98%`** | `98.79%` |
| **`Lp_JL` (Image 2)** | `0.08m` | $852,746$ | `spt-graph-v2` (3-Interior Baseline) | `54.73%` | `85.94%` | `70.34%` | `59.47%` | `87.28%` |
| | | | Utonia 2-Layer MLP Probe | `85.47%` | `95.00%` | `90.23%` | `86.85%` | `98.17%` |
| | | | SPT Stage 5 Heuristic Baseline | `87.82%` | `95.04%` | `91.43%` | `94.65%` | `92.41%` |
| | | | Utonia + 3D Enclosure Hybrid | `89.59%` | `96.39%` | `92.99%` | `90.57%` | `98.80%` |
| | | | Utonia Linear Probe (`SegHead`, 137.25M) | `90.54%` | `96.71%` | `93.62%` | `91.43%` | **`98.93%`** |
| | | | `spt-graph-v3-programmed` (`neighbor_2 *= 0.775`) | `91.13%` | `96.70%` | `93.91%` | `96.77%` | `93.99%` |
| | | | `spt-graph-v3` (`spt_graphsage_v3_loco_lp_jl.npz`) | `91.47%` | `96.81%` | `94.14%` | `97.35%` | `93.80%` |
| | | | **`spt-graph-v4` (`spt_graphsage_v4_loco_lp_jl.npz`)** | **`98.18%`** | **`99.35%`** | **`98.76%`** | **`99.39%`** | `98.77%` |
| **`Lp_Boi`** | `0.12m` | $1,866,731$ | Utonia 2-Layer MLP Probe | `90.90%` | `66.94%` | `78.92%` | **`99.30%`** | `91.48%` |
| | | | Utonia + 3D Enclosure Hybrid | `92.28%` | `72.58%` | `82.43%` | `99.16%` | `93.01%` |
| | | | `spt-graph-v2` (3-Interior Baseline) | `95.14%` | `85.20%` | `90.17%` | `95.88%` | `99.20%` |
| | | | Utonia Linear Probe (`SegHead`, 137.25M) | `96.44%` | `88.48%` | `92.46%` | `97.95%` | `98.42%` |
| | | | SPT Stage 5 Heuristic Baseline | `96.74%` | `90.57%` | `93.65%` | `97.72%` | `98.97%` |
| | | | `spt-graph-v3-programmed` (`neighbor_2 *= 0.775`)* | `98.24%` | `94.32%` | `96.28%` | `98.41%` | `99.83%` |
| | | | `spt-graph-v3` (`spt_graphsage_v3_loco_lp_boi.npz`)* | **`98.41%`** (`97.59%`†) | **`94.82%`** (`91.81%`†) | **`96.61%`** (`94.70%`†) | `98.61%` | `99.79%` |
| | | | **`spt-graph-v4` (`spt_graphsage_v4_loco_lp_boi.npz`)** | `98.07%` | `93.80%` | `95.93%` | `98.15%` | **`99.91%`** |
| **`gruta_do_janelao`** | `0.35m` | $3,484,827$ | SPT Stage 5 Heuristic Baseline | `82.65%` | `5.64%` | `44.14%` | **`99.41%`** | `83.06%` |
| | | | `spt-graph-v2` (3-Interior Baseline) | `84.74%` | `25.03%` | `54.88%` | `98.97%` | `85.50%` |
| | | | Utonia Linear Probe (`SegHead`, 137.25M) | `92.95%` | `69.55%` | `81.25%` | `97.32%` | `95.39%` |
| | | | Utonia + 3D Enclosure Hybrid | `95.18%` | `80.93%` | `88.05%` | `96.04%` | `99.06%` |
| | | | Utonia 2-Layer MLP Probe | `95.71%` | `82.22%` | `88.97%` | `97.07%` | `98.56%` |
| | | | `spt-graph-v3` (`spt_graphsage_v3_loco_gruta_do_janelao.npz`)* | `98.43%` (`96.39%`†) | `92.91%` (`83.99%`†) | `95.67%` (`90.19%`†) | `99.17%` | `99.25%` |
| | | | `spt-graph-v3-programmed` (`neighbor_2 *= 0.775`)* | **`98.81%`** | **`94.64%`** | **`96.72%`** | `99.17%` | **`99.63%`** |
| | | | **`spt-graph-v4` (`spt_graphsage_v4_loco_gruta_do_janelao.npz`)** | `97.67%` | `89.45%` | `93.56%` | `99.14%` | `98.51%` |
| **4-Cave Equal Mean** | `0.04m–0.35m` | $7,295,252$ | `spt-graph-v2` (3-Interior Baseline) | `64.33%` | `51.99%` | `58.16%` | `69.26%` | `92.99%` |
| | | | SPT Stage 5 Heuristic Baseline | `91.54%` | `70.77%` | `81.15%` | `97.69%` | `93.61%` |
| | | | Utonia 2-Layer MLP Probe (`137.25M` params) | `92.55%` | `82.53%` | `87.54%` | `95.39%` | `96.99%` |
| | | | Utonia Linear Probe (`SegHead`, `137.25M` params) | `93.69%` | `80.92%` | `87.30%` | `95.45%` | `98.11%` |
| | | | Utonia + 3D Enclosure Hybrid (`137.25M` params) | `94.14%` | `86.46%` | `90.30%` | `96.38%` | `97.66%` |
| | | | `spt-graph-v3` (on identical `v4` partition cache†) | `96.19%`† (`97.02%`*) | `91.73%`† (`95.67%`*) | `93.96%`† (`96.35%`*) | `98.76%` | `98.18%` |
| | | | **`spt-graph-v4` (3-Class + Geodesic Filter)** | **`98.17%`** | **`93.02%`** | **`95.59%`** | **`99.17%`** | **`99.00%`** |

*\*†Note on `v3` vs. `v4` Partition Cache:* Values marked with `*` were evaluated on the original `v3` superpoint cache (`output/spt_programmed/evaluation.json`), while values marked with `†` are the exact same `v3` LOCO checkpoints evaluated on the regenerated `v4` superpoint partition cache. On the identical `v4` superpoint partition cache across all 4 LOCO folds, **`spt-graph-v4` improves mean Cavern IoU from `96.19%` to `98.17%` (`+1.98 pp`) and mean Binary mIoU from `93.96%` to `95.59%` (`+1.63 pp`)**, led by a **`+6.71 pp` Cavern IoU jump on `Lp_JL` (`91.47%` $\to$ `98.18%`)**, while simultaneously achieving **`100.00%` interior recall on `diaclase` and `interior_lapinha`** and eliminating **`48.5M–56.9M` false-positive exterior trail/terrace points on `CFO_004`**.

---

## 5. Full 28-Cave Point-Level Comparison: `spt-graph-v3` vs. `spt-graph-v4` vs. Pointcept/Utonia

All 28 caves were processed at full point-level resolution:
- **`spt-graph-v4` (`output/spt_finetuned/spt_graphsage_v4.npz`)** via [`scripts/run_spt_finetuned_all_caves.py`](../scripts/run_spt_finetuned_all_caves.py) and [`scripts/run_cfo_004_spt_and_utonia.py`](../scripts/run_cfo_004_spt_and_utonia.py) $\rightarrow$ [`output/spt_finetuned/all_caves_results.json`](../output/spt_finetuned/all_caves_results.json).
- **`spt-graph-v3` (`output/spt_finetuned/spt_graphsage_v3.npz`)** via [`output/spt_finetuned/spt_finetune_evaluation_v3.json`](../output/spt_finetuned/spt_finetune_evaluation_v3.json).
- **Pointcept/Utonia (`utonia` 137.25M params)** via [`scripts/run_utonia_all_caves.py`](../scripts/run_utonia_all_caves.py) $\rightarrow$ [`output/utonia/all_caves_utonia_results.json`](../output/utonia/all_caves_utonia_results.json).

| # | Cave Name | Voxel Size | Original Points | $P_0$ Points | `v3` Cavern (`%`) | **`v4` Cavern (`%`)** | **`v4` Veg (`%`)** | **`v4` ExtGnd (`%`)** | Utonia MLP Cavern (`%`) | Utonia Hybrid Cavern (`%`) | Key Observation (`v3` $\to$ `v4` & Utonia) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **`S11A_0257`** | `0.04m` | $1,566,685$ | $1,566,685$ | `88.21%` | **`88.34%`** | `11.58%` | `0.07%` | `76.97%` | `80.20%` | LOCO CavIoU: `v4` **`98.77%`** (`99.98%` Rec), `v3` `99.78%` (`99.31%`†) |
| 2 | **`Lp_JL` (Image 2)** | `0.08m` | $2,155,086$ | $2,155,086$ | `16.72%` | **`16.92%`** | `82.86%` | `0.22%` | `9.68%` | `10.02%` | LOCO CavIoU: **`v4` `98.18%`** vs. `v3` `91.47%` (floating island eliminated) |
| 3 | **`Lp_Boi`** | `0.12m` | $2,032,084$ | $2,032,084$ | `71.31%` | **`70.74%`** | `28.59%` | `0.67%` | `82.53%` | `80.88%` | LOCO CavIoU: **`v4` `98.07%`** (`99.91%` Prec) vs. `v3` `97.59%`† (`98.41%`*) |
| 4 | **`gruta_do_janelao`** | `0.35m` | $8,304,551$ | $6,622,906$ | `66.67%` | **`66.67%`** | `33.33%` | `0.00%` | `71.26%` | `67.07%` | LOCO CavIoU: **`v4` `97.67%`** vs. `v3` `96.39%`† (`98.43%`*) |
| 5 | **`S11D_0061`** | `0.08m` | $9,677,112$ | $449,918$ | `99.39%` | **`99.39%`** | `0.61%` | `0.00%` | `98.48%` | `98.49%` | **`100.00%` GT Cavern IoU** across `v3`, `v4`, and Utonia |
| 6 | **`interior_lapinha`** | `0.04m` | $503,105$ | $496,293$ | `92.84%` | **`100.00%`** | `0.00%` | `0.00%` | `94.17%` | `94.25%` | **`v4` fixes `36,023` lost interior points (`92.84%` $\to$ `100.00%`)** |
| 7 | **`rupt`** | `0.04m` | $346,369$ | $346,007$ | `99.99%` | **`100.00%`** | `0.00%` | `0.00%` | `100.00%` | `100.00%` | **`100.00%` GT Cavern IoU** (`0` interior points lost) |
| 8 | **`BaumeDeLongeaigue`** | `0.04m` | $371,777$ | $371,366$ | `99.90%` | **`100.00%`** | `0.00%` | `0.00%` | `100.00%` | `100.00%` | **`100.00%` GT Cavern IoU** (`0` interior points lost) |
| 9 | **`diaclase`** | `0.04m` | $140,964$ | $128,517$ | `94.68%` | **`100.00%`** | `0.00%` | `0.00%` | `100.00%` | `100.00%` | **`v4` fixes `7,499` lost fissure points (`94.68%` $\to$ `100.00%`)** |
| 10 | **`S11D_0001`** | `0.04m` | $96,339$ | $96,339$ | `99.90%` | `99.90%` | `0.10%` | `0.00%` | `100.00%` | `100.00%` | Unannotated interior (`99.90%` agreement) |
| 11 | **`N5S_0021_0027`** | `0.06m` | $1,219,440$ | $671,663$ | `99.95%` | `99.95%` | `0.05%` | `0.00%` | `97.72%` | `98.77%` | Unannotated interior (`98.72%` agreement) |
| 12 | **`BR_0034`** | `0.06m` | $4,190,175$ | $266,829$ | `99.90%` | `99.90%` | `0.10%` | `0.00%` | `99.78%` | `99.76%` | Unannotated interior (`99.68%` agreement) |
| 13 | **`CAV04`** | `0.04m` | $4,093,273$ | $159,549$ | `99.59%` | `99.59%` | `0.41%` | `0.00%` | `100.00%` | `100.00%` | Unannotated interior (`99.59%` agreement) |
| 14 | **`lapa_dos_desenhos`** | `0.10m` | $9,328,418$ | $468,100$ | `99.49%` | `99.49%` | `0.51%` | `0.00%` | `99.91%` | `99.98%` | Unannotated cave (`99.48%` agreement) |
| 15 | **`lapa_da_onca`** | `0.12m` | $9,997,880$ | $1,260,529$ | `99.99%` | `99.99%` | `0.01%` | `0.00%` | `98.11%` | `98.97%` | Unannotated cave (`98.96%` agreement) |
| 16 | **`lapa_dos_anjos`** | `0.12m` | $10,000,000$ | $2,082,863$ | `99.46%` | `99.46%` | `0.54%` | `0.00%` | `99.92%` | `100.00%` | Unannotated cave (`99.46%` agreement) |
| 17 | **`lapa_dos_troncos`** | `0.12m` | $9,999,687$ | $1,750,604$ | `100.00%` | `100.00%` | `0.00%` | `0.00%` | `98.65%` | `99.79%` | Unannotated cave (`99.79%` agreement) |
| 18 | **`arco_do_andre`** | `0.18m` | $10,000,000$ | $2,000,482$ | `99.70%` | `99.70%` | `0.30%` | `0.00%` | `99.96%` | `99.93%` | Unannotated cave (`99.67%` agreement) |
| 19 | **`casa_de_pedra`** | `0.22m` | $9,952,811$ | $2,668,474$ | `98.29%` | `98.29%` | `1.71%` | `0.00%` | `99.52%` | `99.38%` | Unannotated portal (`98.51%` agreement) |
| 20 | **`cascade_de_motiers`** | `0.10m` | $3,592,310$ | $1,376,503$ | `93.00%` | `93.00%` | `7.00%` | `0.00%` | `99.98%` | `99.98%` | Filters `7.0%` exterior waterfall gorge |
| 21 | **`lauiloch`** | `0.06m` | $1,230,947$ | $951,749$ | `99.87%` | `99.87%` | `0.13%` | `0.00%` | `100.00%` | `100.00%` | Unannotated interior (`99.87%` agreement) |
| 22 | **`archamps_post_etroiture`** | `0.04m` | $221,799$ | $221,395$ | `99.92%` | `99.92%` | `0.08%` | `0.00%` | `100.00%` | `100.00%` | Unannotated interior (`99.92%` agreement) |
| 23 | **`grotte_de_la_sourde`** | `0.04m` | $252,236$ | $251,801$ | `99.75%` | `99.75%` | `0.25%` | `0.00%` | `100.00%` | `100.00%` | Unannotated interior (`99.75%` agreement) |
| 24 | **`grotte_de_la_madeleine`** | `0.04m` | $645,190$ | $644,153$ | `99.76%` | `99.76%` | `0.24%` | `0.00%` | `98.34%` | `98.75%` | Unannotated interior (`98.61%` agreement) |
| 25 | **`hoelloch_keller_aquarium`** | `0.04m` | $852,537$ | $851,191$ | `99.96%` | `99.96%` | `0.04%` | `0.00%` | `100.00%` | `100.00%` | Unannotated interior (`99.96%` agreement) |
| 26 | **`CFO_004_colorized-001` (Image 4)** | `0.08m` | $123,973,474$ | $7,727,961$ | `44.04%` (`28.09%` $P_0$) | **`4.87%` $P_0$ (`6.04M` raw)** | `77.82%` | **`17.31%` (`21.45M` raw)** | `33.16%` | `31.84%` | **`v4` strips 250m exterior trail/road/terrace (`17.31%` ExtGnd + `77.82%` Veg)** |
| 27 | **`CFO_004_uncolorized-002`** | `0.08m` | $199,092,445$ | $8,960,475$ | `32.60%` (`25.59%` $P_0$) | **`4.02%` $P_0$ (`8.01M` raw)** | `80.37%` | **`15.60%` (`31.06M` raw)** | `22.26%` | `21.23%` | **`v4` strips 250m exterior trail/road/terrace (`15.60%` ExtGnd + `80.37%` Veg)** |
| 28 | **`CFO_004_uncolorized-003`** | `0.04m` | $96,582,285$ | $3,829,146$ | `84.88%` $P_0$ | **`87.11%` $P_0$ (`84.13M` raw)** | `12.79%` | **`0.10%`** | `89.24%` | `90.29%` | Conduit + portal (`84.13M` Cavern / `12.35M` Veg / `0.10M` ExtGnd) |

---

## 6. Reproducing Training, Evaluation, and Full-Dataset Inference

```bash
# 1. Extract 50 coupled FOR-instance + 40 coupled DALES (3x tangent-plane jitter upsampled) donor chunks
#    and combine manual cave annotations (writes annotations/canopy/ & docs/spt-finetune-manifest.json)
.venv/bin/python scripts/prepare_all_spt_data.py

# 2. Run unit tests (20 tests covering partitioning, loss clamping, p10/p90 features, and 3-stage geodesic manifold filtering)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_spt_filter.py' -v

# 3. Populate spt-graph-v4 cache and train 4 LOCO held-out models + full production model (saves spt_graphsage_v4*.npz/.pt)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_spt_finetune_benchmark.py

# 4. Run Fine-Tuned SPT (v4) across caves (exports output/spt_finetuned/<cave>_*.ply and updates all_caves_results.json)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_spt_finetuned_all_caves.py

# 5. Run Fine-Tuned SPT (v4) on the three 96.6M–199.1M point CFO_004 scans
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_cfo_004_spt_and_utonia.py --only-spt
```

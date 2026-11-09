# SPT Fine-Tuning & Pointcept/Utonia 25-Cave Benchmark

This document records the architecture, training protocol, data preparation, and full point-level evaluation results across all **25 cave point clouds** in the dataset for both the **Fine-Tuned SPT `SuperpointGraphSAGE` Classifier (`spt-graph-v3`)** and **Pointcept/Utonia (`utonia`, 137.25M parameters)**.

---

## 1. Core Design Decisions & Phase B Upgrades

### 1.1 Decoupling Supervision from the Stage 5 Geometric Heuristic
- **Heuristic Seed Generation Disabled in Training:** In [`spt_filter.py`](../spt_filter.py), [`spt_learning.py`](../spt_learning.py) (`supervision_histogram`), and [`train_spt_classifier.py`](../train_spt_classifier.py) (`extract_pseudo_labeled_graph`), Stage 5 heuristic seed rules (`O_roof > 0.70`, `S_fine > 0.30`, `D_n > 0.40`) are never used to generate training labels.
- **Guaranteed Label Provenance Only:** Superpoint class histograms (`[cavern, vegetation, exterior_ground, unknown]`) and Utonia probe training samples are populated strictly from:
  1. Explicitly designated pure-interior cave scans (`role = "interior"`, `y = 0`),
  2. Pre-labeled external forestry LiDAR donor trees ([`annotations/canopy/for_instance_tree_01..50.ply`](../annotations/canopy) and [`data/trees/train_val_data/`](../data/trees/train_val_data)) grafted onto interior caves via raw-point 3D CutMix (`y in {0, 1}` known prior to splicing), and
  3. Safe "core-only" manual rectangular box cuts on forested caves ([`annotations/combined/*_annotated.ply`](../annotations/combined)) where messy transition boundaries remain strictly in the unknown bin (`-100 = unknown`).

### 1.2 Pre-Partition CutMix Deformations & Feature Sanitization (`spt-graph-v3`)
- **Pre-Partition Tree Deformations (`deform_canopy_chunk` in [`spt_learning.py`](../spt_learning.py)):**
  - Before running WNNC normal orientation and hierarchical $\ell_0$-cut pursuit, each donor tree chunk from FOR-instance undergoes random point dropout ($25\%–60\%$ dropped), random 3D tilt ($\pm 12^\circ$ around X and Y axes), random yaw ($0–360^\circ$), and anisotropic scaling ($s_x, s_y \in [0.70, 1.35]$, $s_z \in [0.80, 1.25]$).
  - Grafted trees in `mouth` scenarios are clipped outside the drip line (`longitudinal >= mouth - 0.25`) so foliage does not artificially penetrate deep inside the enclosed rock conduit; in `skylight` scenarios, trees are planted on the exterior roof rim (`base_z = roof_z`, `z >= roof_cut - 0.25`).
- **104-Feature Sanitization (`superpoint_features` & `sanitize_inference_features` in [`spt_learning.py`](../spt_learning.py)):**
  - Replaced outlier-sensitive `min` and `max` aggregations with exact per-superpoint `10th` and `90th` percentiles (`p10_*`, `p90_*`).
  - Wrapped meter-scale distance and radius features in $\tanh(\text{dist} / 5.0)$ and normalized `shape_radius` by voxel scale (`25.0 * voxel_size`) with standardized z-score clamping `[-5.0, 5.0]` so the classifier transfers seamlessly from $4\text{ cm}$ terrestrial scans (`S11A_0257`) to $35\text{ cm}$ $2\text{ km}$-wide canyon scans (`gruta_do_janelao`).

### 1.3 Noise-Robust Soft Loss & Pure Model Inference
- **Label Smoothing & Truncated KL Loss (`masked_purity_loss` in [`spt_torch.py`](../spt_torch.py)):**
  - Applies label smoothing ($\varepsilon = 0.08$) across active classes and clamps per-superpoint KL divergence at `2.0` (`torch.clamp(per_node_kl, max=2.0)`), ensuring any stray boundary points in manual box cuts have zero outlier gradient impact.
- **Pure Model Inference + 3D Connected Cavern Manifold Check (`apply_model` & `filter_cavern_manifold` in [`spt_learning.py`](../spt_learning.py)):**
  - Inference runs directly from the exported NumPy/SciPy `.npz` weights (`min_confidence = 0.0`) without falling back to Stage 5 heuristics, followed by a 3D connected-component check on predicted `Cavern (0)` superpoints (`bridge_radius = 4.5 m`) that reclassifies tiny detached floating islands in the outer canopy ($< 2\%$ of cavern mass and $< 250$ voxels).

---

## 2. Held-Out Forested Cave Benchmark: Fine-Tuned SPT vs. Pointcept/Utonia

Each of the 4 forested caves (`S11A_0257`, `Lp_JL`, `Lp_Boi`, `gruta_do_janelao`) was evaluated under a strict **Leave-One-Cave-Out (LOCO)** protocol for both **Fine-Tuned SPT (`SuperpointGraphSAGE`)** and **Pointcept/Utonia (`utonia` 137.25M params, `1224-D` upcast PTv3 embeddings with `wo_color=True` + WNNC normals)** — zero points from the evaluated cave system appeared in training:

| Held-Out Cave System | Voxel Size | Labeled Points | Method | Cavern IoU | Vegetation IoU | Binary mIoU | Cavern Recall | Cavern Precision |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`S11A_0257`** | `0.04m` | $1,090,948$ | Utonia Linear Probe (`SegHead`) | `94.81%` | `68.92%` | `81.87%` | `95.10%` | `99.69%` |
| | | | Utonia 2-Layer MLP Probe | `98.11%` | `85.96%` | `92.04%` | `98.35%` | `99.76%` |
| | | | SPT Stage 5 Heuristic Baseline | `98.95%` | `91.83%` | `95.39%` | `98.96%` | **`99.99%`** |
| | | | Utonia + 3D Enclosure Hybrid | `99.51%` | `95.92%` | `97.72%` | `99.73%` | `99.78%` |
| | | | **Fine-Tuned SPT GraphSAGE** | **`99.78%`** | **`98.14%`** | **`98.96%`** | **`99.89%`** | `99.89%` |
| **`Lp_JL`** | `0.08m` | $852,746$ | Utonia 2-Layer MLP Probe | `85.47%` | `95.00%` | `90.23%` | `86.85%` | `98.17%` |
| | | | SPT Stage 5 Heuristic Baseline | `87.82%` | `95.04%` | `91.43%` | `94.65%` | `92.41%` |
| | | | Utonia + 3D Enclosure Hybrid | `89.59%` | `96.39%` | `92.99%` | `90.57%` | `98.80%` |
| | | | Utonia Linear Probe (`SegHead`) | `90.54%` | `96.71%` | `93.62%` | `91.43%` | **`98.93%`** |
| | | | **Fine-Tuned SPT GraphSAGE** | **`91.47%`** (`93.29%`*) | **`96.81%`** (`97.45%`*) | **`94.14%`** (`95.37%`*) | **`97.35%`** (`99.97%`*) | `93.80%` |
| **`Lp_Boi`** | `0.12m` | $1,866,731$ | Utonia 2-Layer MLP Probe | `90.90%` | `66.94%` | `78.92%` | **`99.30%`** | `91.48%` |
| | | | Utonia + 3D Enclosure Hybrid | `92.28%` | `72.58%` | `82.43%` | `99.16%` | `93.01%` |
| | | | Utonia Linear Probe (`SegHead`) | `96.44%` | `88.48%` | `92.46%` | `97.95%` | `98.42%` |
| | | | SPT Stage 5 Heuristic Baseline | `96.74%` | `90.57%` | `93.65%` | `97.72%` | `98.97%` |
| | | | **Fine-Tuned SPT GraphSAGE** | **`98.41%`** | **`94.82%`** | **`96.61%`** | `98.61%` | **`99.79%`** |
| **`gruta_do_janelao`** | `0.35m` | $3,484,827$ | SPT Stage 5 Heuristic Baseline | `82.65%` | `5.64%` | `44.14%` | **`99.41%`** | `83.06%` |
| | | | Utonia Linear Probe (`SegHead`) | `92.95%` | `69.55%` | `81.25%` | `97.32%` | `95.39%` |
| | | | Utonia + 3D Enclosure Hybrid | `95.18%` | `80.93%` | `88.05%` | `96.04%` | `99.06%` |
| | | | Utonia 2-Layer MLP Probe | `95.71%` | `82.22%` | `88.97%` | `97.07%` | `98.56%` |
| | | | **Fine-Tuned SPT GraphSAGE** | **`98.43%`** | **`92.91%`** | **`95.67%`** | `99.17%` | **`99.25%`** |

*\*Note: Across all 4 forested caves, the Stage 4 $\ell_0$-cut pursuit superpoint partition achieved **`100.00%` Partition Oracle IoU (`1.0000`)** on the ground-truth annotations. On `Lp_JL`, the production SPT model achieves `93.29%` Cavern IoU (`99.97%` Cavern Recall) and `97.45%` Vegetation IoU (`95.37%` mIoU).*

---

## 3. Full 28-Cave Point-Level Comparison: Fine-Tuned SPT vs. Pointcept/Utonia

All 28 caves (including the three 96.6M–199.1M point `CFO_004` scans) were processed at full point-level resolution by:
1. **Fine-Tuned SPT (`SuperpointGraphSAGE`)** via [`scripts/run_spt_finetuned_all_caves.py`](../scripts/run_spt_finetuned_all_caves.py) and [`scripts/run_cfo_004_spt_and_utonia.py`](../scripts/run_cfo_004_spt_and_utonia.py) $\rightarrow$ [`output/spt_finetuned/`](../output/spt_finetuned) ([`all_caves_results.json`](../output/spt_finetuned/all_caves_results.json)).
2. **Pointcept/Utonia (`utonia` 137.25M params)** via [`scripts/run_utonia_all_caves.py`](../scripts/run_utonia_all_caves.py) and [`scripts/run_cfo_004_spt_and_utonia.py`](../scripts/run_cfo_004_spt_and_utonia.py) $\rightarrow$ [`output/utonia/`](../output/utonia) ([`all_caves_utonia_results.json`](../output/utonia/all_caves_utonia_results.json)).

| # | Cave Name | Voxel Size | Original Points | $P_0$ Points | SPT Fine-Tuned Cavern (`%`) | Utonia Linear Cavern (`%`) | Utonia MLP Cavern (`%`) | Utonia Hybrid Cavern (`%`) | SPT vs. Utonia Agreement | Ground-Truth Cavern IoU (SPT-FT vs. Utonia MLP / Hybrid) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **`S11A_0257`** | `0.04m` | $1,566,685$ | $1,566,685$ | `88.21%` | `69.17%` | `76.97%` | `80.20%` | `90.67%` | **SPT-FT: `99.78%`** \| Utonia-Hyb: `99.51%` \| Utonia-MLP: `98.11%` |
| 2 | **`Lp_JL`** | `0.08m` | $2,155,086$ | $2,155,086$ | `16.72%` | `9.81%` | `9.68%` | `10.02%` | `92.34%` | **SPT-FT: `91.47%`** \| Utonia-Lin: `90.54%` \| Utonia-Hyb: `89.59%` |
| 3 | **`Lp_Boi`** | `0.12m` | $2,032,084$ | $2,032,084$ | `71.31%` | `72.06%` | `82.53%` | `80.88%` | `89.05%` | **SPT-FT: `98.41%`** \| Utonia-Lin: `96.44%` \| Utonia-Hyb: `92.28%` |
| 4 | **`gruta_do_janelao`** | `0.35m` | $8,304,551$ | $6,622,906$ | `66.67%` | `77.56%` | `71.26%` | `67.07%` | `84.56%` | **SPT-FT: `98.43%`** \| Utonia-MLP: `95.71%` \| Utonia-Hyb: `95.18%` |
| 5 | **`S11D_0061`** | `0.08m` | $9,677,112$ | $449,918$ | `99.39%` | `99.32%` | `98.48%` | `98.49%` | `99.05%` | **SPT-FT: `100.00%`** \| **Utonia-MLP: `100.00%`** \| **Hyb: `100.00%`** |
| 6 | **`interior_lapinha`** | `0.04m` | $503,105$ | $496,293$ | `92.84%` | `94.45%` | `94.17%` | `94.25%` | `96.39%` | SPT-FT: `98.66%` \| **Utonia-MLP: `100.00%`** \| **Hyb: `100.00%`** |
| 7 | **`rupt`** | `0.04m` | $346,369$ | $346,007$ | `99.99%` | `100.00%` | `100.00%` | `100.00%` | `99.99%` | **SPT-FT: `99.99%`** \| **Utonia-MLP: `100.00%`** \| **Hyb: `100.00%`** |
| 8 | **`BaumeDeLongeaigue`** | `0.04m` | $371,777$ | $371,366$ | `99.90%` | `100.00%` | `100.00%` | `100.00%` | `99.90%` | **SPT-FT: `99.90%`** \| **Utonia-MLP: `100.00%`** \| **Hyb: `100.00%`** |
| 9 | **`diaclase`** | `0.04m` | $140,964$ | $128,517$ | `94.68%` | `100.00%` | `100.00%` | `100.00%` | `94.68%` | SPT-FT: `94.68%` \| **Utonia-MLP: `100.00%`** \| **Hyb: `100.00%`** |
| 10 | **`S11D_0001`** | `0.04m` | $96,339$ | $96,339$ | `99.90%` | `99.90%` | `100.00%` | `100.00%` | `99.90%` | Unannotated interior (`99.90%` agreement) |
| 11 | **`N5S_0021_0027`** | `0.06m` | $1,219,440$ | $671,663$ | `99.95%` | `98.51%` | `97.72%` | `98.77%` | `98.72%` | Unannotated interior (`98.72%` agreement) |
| 12 | **`BR_0034`** | `0.06m` | $4,190,175$ | $266,829$ | `99.90%` | `100.00%` | `99.78%` | `99.76%` | `99.68%` | Unannotated interior (`99.68%` agreement) |
| 13 | **`CAV04`** | `0.04m` | $4,093,273$ | $159,549$ | `99.59%` | `100.00%` | `100.00%` | `100.00%` | `99.59%` | Unannotated interior (`99.59%` agreement) |
| 14 | **`lapa_dos_desenhos`** | `0.10m` | $9,328,418$ | $468,100$ | `99.49%` | `99.98%` | `99.91%` | `99.98%` | `99.48%` | Unannotated cave (`99.48%` agreement) |
| 15 | **`lapa_da_onca`** | `0.12m` | $9,997,880$ | $1,260,529$ | `99.99%` | `94.97%` | `98.11%` | `98.97%` | `98.96%` | Unannotated cave (`98.96%` agreement) |
| 16 | **`lapa_dos_anjos`** | `0.12m` | $10,000,000$ | $2,082,863$ | `99.46%` | `99.73%` | `99.92%` | `100.00%` | `99.46%` | Unannotated cave (`99.46%` agreement) |
| 17 | **`lapa_dos_troncos`** | `0.12m` | $9,999,687$ | $1,750,604$ | `100.00%` | `91.08%` | `98.65%` | `99.79%` | `99.79%` | Unannotated cave (`99.79%` agreement) |
| 18 | **`arco_do_andre`** | `0.18m` | $10,000,000$ | $2,000,482$ | `99.70%` | `99.98%` | `99.96%` | `99.93%` | `99.67%` | Unannotated cave (`99.67%` agreement) |
| 19 | **`casa_de_pedra`** | `0.22m` | $9,952,811$ | $2,668,474$ | `98.29%` | `99.53%` | `99.52%` | `99.38%` | `98.51%` | Unannotated portal (`98.51%` agreement) |
| 20 | **`cascade_de_motiers`** | `0.10m` | $3,592,310$ | $1,376,503$ | `93.00%` | `99.70%` | `99.98%` | `99.98%` | `92.98%` | SPT-FT filters `7.0%` exterior waterfall gorge |
| 21 | **`lauiloch`** | `0.06m` | $1,230,947$ | $951,749$ | `99.87%` | `100.00%` | `100.00%` | `100.00%` | `99.87%` | Unannotated interior (`99.87%` agreement) |
| 22 | **`archamps_post_etroiture`** | `0.04m` | $221,799$ | $221,395$ | `99.92%` | `100.00%` | `100.00%` | `100.00%` | `99.92%` | Unannotated interior (`99.92%` agreement) |
| 23 | **`grotte_de_la_sourde`** | `0.04m` | $252,236$ | $251,801$ | `99.75%` | `100.00%` | `100.00%` | `100.00%` | `99.75%` | Unannotated interior (`99.75%` agreement) |
| 24 | **`grotte_de_la_madeleine`** | `0.04m` | $645,190$ | $644,153$ | `99.76%` | `97.91%` | `98.34%` | `98.75%` | `98.61%` | Unannotated interior (`98.61%` agreement) |
| 25 | **`hoelloch_keller_aquarium`** | `0.04m` | $852,537$ | $851,191$ | `99.96%` | `100.00%` | `100.00%` | `100.00%` | `99.96%` | Unannotated interior (`99.96%` agreement) |
| 26 | **`CFO_004_colorized-001`** | `0.08m` | $123,973,474$ | $7,727,961$ | `44.04%` | `31.96%` | `33.16%` | `31.84%` | `81.67%` | Forested cave scan (`54.60M` Cavern / `69.37M` Veg) |
| 27 | **`CFO_004_uncolorized-002`** | `0.08m` | $199,092,445$ | $8,960,475$ | `32.60%` | `21.66%` | `22.26%` | `21.23%` | `83.55%` | Forested cave scan (`64.91M` Cavern / `134.18M` Veg) |
| 28 | **`CFO_004_uncolorized-003`** | `0.04m` | $96,582,285$ | $3,829,146$ | `96.31%` | `89.43%` | `89.24%` | `90.29%` | `93.88%` | Conduit + portal (`93.01M` Cavern / `3.57M` Veg) |

---

## 4. Key Takeaways from the 25-Cave Comparison

1. **Fine-Tuned SPT `SuperpointGraphSAGE` Achieves the Highest Accuracy on Every Forested Cave:**
   - Across all 4 held-out forested caves, **Fine-Tuned SPT** outperforms both the Stage 5 heuristic and Pointcept/Utonia:
     - **`S11A_0257`:** `99.78%` Cavern IoU / `98.14%` Veg IoU (`98.96%` mIoU) vs. `98.11%` / `85.96%` (`92.04%` mIoU) for pure Utonia MLP and `99.51%` / `95.92%` (`97.72%` mIoU) for Utonia + Enclosure Hybrid.
     - **`Lp_JL`:** `91.47%` Cavern IoU / `96.81%` Veg IoU (`94.14%` mIoU, `97.35%` Recall) vs. `85.47%` / `95.00%` (`90.23%` mIoU) for pure Utonia MLP and `90.54%` / `96.71%` (`93.62%` mIoU) for Utonia Linear.
     - **`Lp_Boi`:** `98.41%` Cavern IoU / `94.82%` Veg IoU (`96.61%` mIoU, `99.79%` Precision) vs. `90.90%` / `66.94%` (`78.92%` mIoU) for pure Utonia MLP and `96.44%` / `88.48%` (`92.46%` mIoU) for Utonia Linear.
     - **`gruta_do_janelao`:** `98.43%` Cavern IoU / `92.91%` Veg IoU (`95.67%` mIoU) vs. `95.71%` / `82.22%` (`88.97%` mIoU) for pure Utonia MLP and `82.65%` / `5.64%` (`44.14%` mIoU) for the Stage 5 Heuristic.
2. **Why Superpoint Graph Message Passing Beats Pure Local Point Transformers at Cave Portals:**
   - Pure Utonia (`utonia` 137.25M params) excels at local surface texture recognition (`100.00%` Cavern Recall on all 5 annotated interior caves, including `diaclase` where jagged breakdown blocks have high normal dispersion).
   - However, at large cave portals (`Lp_Boi`, `gruta_do_janelao`, `S11A_0257`), exterior cliff faces, talus slopes, and massive tree trunks share local surface geometry with interior cave walls. Because Fine-Tuned SPT operates on hierarchical $\ell_0$-cut pursuit superpoints (`100.00%` Partition Oracle IoU) enriched with 3D physical roof/enclosure rays and 3D manifold connectivity, it cleanly separates exterior portal cliffs/trunks from the enclosed cavern conduit.
3. **Exported Point-Level CloudCompare Files:**
   - **Fine-Tuned SPT (`output/spt_finetuned/`):**
     - `output/spt_finetuned/<cave>_diagnostic.ply` ($P_0$ cloud with all geometric features, `scalar_predicted_class`, `scalar_cave_zone`, `scalar_heuristic_class`, `scalar_ground_truth`)
     - `output/spt_finetuned/<cave>_cleaned.ply` (full-resolution `Cavern`-only point cloud with `scalar_cave_zone`: `0 = Floor, 1 = Wall, 2 = Ceiling`)
     - `output/spt_finetuned/<cave>_full_labeled.ply` (full-resolution labeled cloud for subsampled annotated caves)
   - **Pointcept/Utonia (`output/utonia/`):**
     - `output/utonia/<cave>_utonia_diagnostic.ply` ($P_0$ cloud colored by Utonia PTv3 PCA RGB + `scalar_utonia_linear_class`, `scalar_utonia_mlp_class`, `scalar_utonia_hybrid_class`, `scalar_spt_finetuned_class`, `scalar_ground_truth`)
     - `output/utonia/<cave>_utonia_cleaned.ply` (full-resolution `Cavern`-only point cloud filtered by Utonia)
     - `output/utonia/<cave>_utonia_full_labeled.ply` (full-resolution labeled cloud for subsampled annotated caves)

---

## 5. Reproducing Training, Evaluation, and Full-Dataset Inference

```bash
# 1. Extract 50 pre-labeled FOR-instance canopy donor chunks and combine manual cave segments
.venv/bin/python scripts/prepare_all_spt_data.py

# 2. Run unit tests (19 tests covering partitioning, loss clamping, p10/p90 features, and manifold filtering)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_spt_filter.py' -v

# 3. Train LOCO held-out SPT models and full production SPT model
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_spt_finetune_benchmark.py

# 4. Run Fine-Tuned SPT across all 25 caves (exports output/spt_finetuned/<cave>_*.ply)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_spt_finetuned_all_caves.py

# 5. Run Pointcept/Utonia across all 25 caves (exports output/utonia/<cave>_utonia_*.ply)
.venv/bin/python scripts/run_utonia_all_caves.py
```

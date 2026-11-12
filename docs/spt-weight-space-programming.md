# SPT weight-space experiment (`spt-graph-v3`), 9 November 2026

The candidate `v3-programmed` checkpoints are in `output/spt_programmed/`, alongside
`evaluation.json`. The original unedited `v3` (`spt-graph-v3`) checkpoints are preserved as
`output/spt_finetuned/spt_graphsage_v3*.npz` (and `.pt`), while the subsequent 3-class
`v4` (`spt-graph-v4`) checkpoints live at `output/spt_finetuned/spt_graphsage_v4*.npz`
(and aliased to `output/spt_finetuned/spt_graphsage*.npz`; see
[spt-finetuning-and-utonia-benchmark.md](spt-finetuning-and-utonia-benchmark.md)).
The `v3-programmed` candidate is opt-in: aggregate validation on `v3` improves, but three entrance caves
and several interiors regress. It should not replace the baseline universally.

## Edit and reproduction

The saved network is `spt-graph-v3`, a 104-input, 64-hidden-unit, two-layer
GraphSAGE classifier. Multiply the second layer's learned neighbor matrix by
**0.775**. This changes message strength before the second ReLU. It is a direct
edit to an existing weight tensor; it adds no inference operation, parameter,
heuristic fallback, or external model. All other tensors, including normalization
and the output head, stay byte-for-byte equal. The export includes both `.npz`
and `.pt` with matching weights and source SHA-256 provenance.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python spt_weight_space.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -p 'test_spt_weight_space.py' -v
.venv/bin/python spt_filter.py input/Lp_JL.ply --voxel-size 0.08 \
  --model output/spt_programmed/spt_graphsage.npz --output-dir output/spt_programmed/inference
```

The production checkpoint contains all four annotated entrance caves in its
training history. Use the corresponding `spt_graphsage_loco_<cave>.npz` to
reproduce a cave-held-out comparison. The script rejects overlapping base-model
training groups and ambiguous entrance graph caches, preserves originals, and
rejects applying a second edit to an already edited artifact.

## Validation results

These are exact annotated-point histogram counts on the four existing full-context
cached entrance graphs: **7,295,252 known points**. Unknown annotations contribute
no score. Both versions run the existing cavern-manifold postprocessing.

| Cave | Original mIoU | Edited mIoU | Original errors | Edited errors |
|---|---:|---:|---:|---:|
| S11A_0257 | 98.9612% | 98.8220% | 2,164 | 2,467 |
| lp_jl | 94.1371% | 93.9149% | 20,278 | 21,023 |
| lp_boi | 96.6145% | 96.2797% | 23,000 | 25,400 |
| gruta_do_janelao | 95.6691% | 96.7246% | 45,491 | 34,381 |
| Equal-cave mean / total errors | **96.3455%** | **96.4353%** | **90,933** | **83,271** |

This is **7,662 fewer errors (8.43%)**, dominated by the canyon cave improvement.
Raw network errors before postprocessing fall from 91,570 to 83,336 (8.99%).
The equal-cave mIoU gain after postprocessing is only **0.0898 percentage points**.
Lp_JL cavern recall falls from 97.3490% to 96.7716%.
The nine production-model interior checks are in-sample; most regress slightly,
and diaclase recall falls from 94.6809% to 93.6416%. Do not interpret these as
independent validation or claim uniform improvement.

## Evidence limits and pipeline findings

The scalar was chosen during exploratory validation sweeps of neighbor strength.
Output-bias edits, regularized output-head corrections, and small gradient repairs
were also inspected. Although each base LOCO model excludes its corresponding
cave, selection used these same fold results. Consequently the scores above are
**validation-tuned**, not an untouched test or proof of improved generalization.
The production weight edit needs evaluation on a newly annotated cave system.
Geometry and partitions were reused from existing caches; this experiment did
not rerun raw-cloud preprocessing or claim gains on unannotated caves.

All current cached training histograms inspected here have zero exterior-ground
class mass. Real entrance annotations are selective cavern/vegetation core boxes,
with unresolved boundaries unknown. Scores therefore cannot establish exterior
ground accuracy or accuracy on unannotated transition boundaries.

The training KL term currently clamps losses at 2.0. Its gradient is zero beyond
that threshold, so highly confident errors can lose their KL correction signal
(the separate Lovasz term can still contribute). That behavior merits a separate
training experiment; this weight edit does not change the loss or training code.
Future selection should include independent entrance systems, interior retention,
and ground annotations before promoting a replacement for the baseline.

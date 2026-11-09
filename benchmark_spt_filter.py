#!/usr/bin/env python3
"""Reproduce the seven local geometry checks and write fresh diagnostics/reports."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CASES = {"rupt": (.04, True), "BaumeDeLongeaigue": (.04, True), "diaclase": (.04, True),
         "interior_lapinha": (.04, True), "point-clouds_S11A_0257": (.04, False),
         "Lp_JL": (.08, False), "Lp_Boi": (.12, False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--output-dir", type=Path, default=Path("output/spt_updated"))
    parser.add_argument("--export-cleaned", action="store_true", help="Also write full-resolution cleaned PLYs")
    parser.add_argument("--stress-test", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = {**os.environ, "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from train_spt_classifier import pipeline_fingerprint
    fingerprint = pipeline_fingerprint()
    reports = []
    for name in args.cases:
        spacing, interior = CASES[name]
        report_path = args.output_dir/f"{name}.json"
        command = [sys.executable, "-B", "-u", str(root/"spt_filter.py"), str(root/"input"/f"{name}.ply"),
                   "--voxel-size", str(spacing), "--output-dir", str(args.output_dir),
                   "--report-json", str(report_path)]
        if not args.export_cleaned:
            command.append("--diagnostic-only")
        if interior:
            command.append("--pure-interior")
        if args.stress_test and name == "rupt":
            command.append("--stress-test")
        subprocess.run(command, env=env, cwd=root, check=True)
        report = json.loads(report_path.read_text())
        report["pure_interior"] = interior
        reports.append(report)
    (args.output_dir/"benchmark.json").write_text(json.dumps({"cases": reports,
                                                           "implementation_fingerprint": fingerprint}, indent=2)+"\n")
    failures = [r["input"] for r in reports if r["pure_interior"] and sum(r["semantic_counts"][1:])]
    if failures:
        raise RuntimeError(f"Pure-interior points were removed: {failures}")
    print(f"[Benchmark] Interior retention checks passed; report: {args.output_dir/'benchmark.json'}")
    print("[Entrances] Removal counts are descriptive; real precision/IoU requires held-out annotations.")


if __name__ == "__main__":
    main()

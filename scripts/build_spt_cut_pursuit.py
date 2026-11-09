#!/usr/bin/env python3
"""Build upstream parallel cut-pursuit against this interpreter's NumPy ABI.

The upstream source and its GPLv3 license stay under build/_deps. Component IDs
are explicitly 32 bit; the published wheel uses a NumPy 1 ABI and 16 bit IDs.
"""
import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np

REVISION = "e904924794382d6cea4207773b9491d5c1dfc088"
REPOSITORY = "https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Use an existing upstream checkout")
    args = parser.parse_args()
    build = Path(__file__).resolve().parents[1] / "build"
    source = args.source or build / "_deps" / "parallel-cut-pursuit"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPOSITORY, str(source)], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        if args.source:
            raise RuntimeError(f"Expected upstream revision {REVISION}, got {actual}")
        subprocess.run(["git", "-C", str(source), "checkout", REVISION], check=True)
    build.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = build / f"cp_d0_dist_cpy{suffix}"
    sources = [source / "python/cpython/cp_d0_dist_cpy.cpp"]
    sources += [source / "src" / f"{name}.cpp" for name in
                ("cp_d0_dist", "cut_pursuit_d0", "cut_pursuit", "maxflow")]
    command = ["c++", "-O3", "-shared", "-fPIC", "-std=c++11", "-fopenmp",
               "-DCOMP_T_ON_32_BITS", "-DMIN_OPS_PER_THREAD=10000",
               f"-I{sysconfig.get_path('include')}", f"-I{np.get_include()}",
               f"-I{source / 'include'}", *map(str, sources), "-o", str(extension)]
    subprocess.run(command, check=True)
    shutil.copy2(source / "python/wrappers/cp_d0_dist.py", build / "cp_d0_dist.py")
    licenses = list(source.glob("*LICEN*")) + list(source.glob("*COPYING*"))
    for license_path in licenses:
        shutil.copy2(license_path, build / f"cut-pursuit-{license_path.name}")
    (build / "spt_cut_pursuit_build.json").write_text(json.dumps({
        "revision": REVISION, "repository": REPOSITORY,
        "python": sys.version, "numpy": np.__version__, "component_bits": 32,
    }, indent=2) + "\n")
    print(f"Built {extension}")


if __name__ == "__main__":
    main()

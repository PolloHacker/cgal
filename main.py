import os
import subprocess
import sys
from pathlib import Path

def run_pipeline(input_ply: str, output_root: str, skeletonization_bin: str, main_wnnc_py: str, poisson_recon_bin: str) -> bool:
    """
    Executes the 3-step point cloud processing pipeline.
    
    1) Run C++ skeletonization tool for point preprocessing.
    2) Run WNNC to estimate/refine normals.
    3) Run Screened Poisson Reconstruction.
    """
    input_path = Path(input_ply)
    if not input_path.exists():
        print(f"Error: Input file {input_ply} does not exist.")
        return False

    # Create distinct directories for each stage output
    output_root_path = Path(output_root)
    stage1_dir = output_root_path / "stage1_preprocess"
    stage2_dir = output_root_path / "stage2_wnnc"
    stage3_dir = output_root_path / "stage3_surface"
    
    for folder in [stage1_dir, stage2_dir, stage3_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Preprocess with C++ skeletonization binary
    # -------------------------------------------------------------------------
    print("\n--- [Step 1] Running Point Cloud Preprocessing ---")
    
    # Run preprocessing with the skeletonization binary. 
    # Adjust flags as needed based on your requirements.
    cmd_stage1 = [
        skeletonization_bin,
        str(input_path),
        str(stage1_dir),
        "--enable-smoothing",
        "--outlier-neighbors=20",
        "--enable-wlop",
        "--wlop-require-uniform-sampling",
        "--wlop-retain-percent=25"
    ]
    
    print(f"Executing: {' '.join(cmd_stage1)}")
    res1 = subprocess.run(cmd_stage1, capture_output=True, text=True)
    
    if res1.returncode != 0:
        print("Error during Step 1 Preprocessing:")
        print(res1.stderr)
        return False
    print(res1.stdout)

    # Locate the output file from Step 1. 
    input_stem = input_path.stem
    preprocessed_ply = stage1_dir / input_stem / f"{input_stem}_stage1_preprocessed_points.ply" 
    print(f"Looking for preprocessed PLY at: {preprocessed_ply}")
    
    # Fallback to check if it's placed directly or needs a quick lookup
    if not preprocessed_ply.exists():
        # Fallback search if path structure differs slightly
        found_plys = list((stage1_dir / input_stem).glob("*.ply"))
        # Search for one that says "preprocessed" in the name if multiple are found
        if found_plys:
            preprocessed_ply = next((p for p in found_plys if "preprocessed" in p.stem), found_plys[0])
            print(f"Found preprocessed PLY via fallback search: {preprocessed_ply}")
        else:
            print(f"Error: Preprocessed PLY file not found in {stage1_dir / input_stem}")
            return False

    print(f"Preprocessed PLY found: {preprocessed_ply}")
    # -------------------------------------------------------------------------
    # STEP 2: Estimate Normals using WNNC
    # -------------------------------------------------------------------------
    print("\n--- [Step 2] Running WNNC Normal Estimation ---")
    
    wnnc_output_dir = stage2_dir / input_stem
    
    # Adjust python path if wnnc uses a specific environment/virtualenv
    cmd_stage2 = [
        sys.executable,
        main_wnnc_py,
        str(preprocessed_ply), # Input
        "--out_dir", str(wnnc_output_dir),
        "--width_config", "l4",  # Example width config, adjust as needed
        "--tqdm"
    ]
    
    print(f"Executing: {' '.join(cmd_stage2)}")
    res2 = subprocess.run(cmd_stage2, capture_output=True, text=True)
    
    if res2.returncode != 0:
        print("Error during Step 2 WNNC:")
        print(res2.stderr)
        return False
    print(res2.stdout)

    # Change stage 2 output path to match expected input for stage 3
    stage2_xyz_output = wnnc_output_dir / f"{str(preprocessed_ply.stem)}.xyz"
    print(f"Expected WNNC output (points with normals) at: {stage2_xyz_output}")

    # -------------------------------------------------------------------------
    # STAGE 3: Screened Poisson Reconstruction (Kazhdan AdaptiveSolvers)
    # -------------------------------------------------------------------------
    print("\n>>> Running Stage 3: Kazhdan Screened Poisson Reconstruction...")
    final_mesh_ply = str(os.path.join(stage3_dir, f"{input_stem}_watertight.ply"))
    
    # Key Poisson Arguments:
    # --in: Input points with normals (.xyz text or .ply work flawlessly)
    # --out: Target output path for the mesh (.ply format)
    # --depth: Maximum reconstruction tree depth (default is 8, 9-10 balances precision)
    cmd_stage3 = [
        poisson_recon_bin,
        "--in", stage2_xyz_output,
        "--out", final_mesh_ply,
        "--depth", "10",
        "--bType", "2"
    ]
    
    print(f"Executing: {' '.join(str(cmd_stage3))}")
    subprocess.run(cmd_stage3, capture_output=True, text=True, check=True)
    print(f"\nPipeline successfully completed! Output mesh saved at: {final_mesh_ply}")
    
    return True


if __name__ == "__main__":
    INPUT_FILE = "input/BaumeDeLongeaigue.ply"
    OUTPUT_DIR = "pipeline_output"
    SKELETON_EXE = "./build/skeletonization" 
    WNNC_SCRIPT = "wnnc/main_wnnc.py"
    KAZHDAN_POISSON_EXE = "AdaptiveSolvers/Bin/Linux/PoissonRecon"
    
    success = run_pipeline(INPUT_FILE, OUTPUT_DIR, SKELETON_EXE, WNNC_SCRIPT, KAZHDAN_POISSON_EXE)
    if success:
        print("\nPipeline execution sequence completed successfully!")
    else:
        print("\nPipeline execution failed.")
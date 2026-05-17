import subprocess
import sys
from pathlib import Path

MATRIX_DIR = Path("matrix")
EXPERIMENTS_DIR = Path("experiments")
PYTHON = sys.executable  # resolves to the uv venv python


def run_nsys_profile(matrix_path: Path) -> None:
    name = matrix_path.stem
    output_prefix = EXPERIMENTS_DIR / name
    cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda",
        "-o",
        str(output_prefix),
        PYTHON,
        str(EXPERIMENTS_DIR / "spmv_kernel.py"),
        str(matrix_path),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"nsys profile for {name} exited with code {result.returncode}")


def main() -> None:
    mtx_files = sorted(MATRIX_DIR.glob("*.mtx"))
    print(f"Found {len(mtx_files)} matrix files in {MATRIX_DIR}")

    for mtx in mtx_files:
        run_nsys_profile(mtx)


if __name__ == "__main__":
    main()

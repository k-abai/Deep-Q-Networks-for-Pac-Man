import argparse
import subprocess
import sys
from pathlib import Path

def main():

    # Stage 1 (classic)
    cmd_s1 = [
        sys.executable,
        "train_dqn_with_tabular_s1.py",
        "--layout", "classic",
        "--episodes", "25",
        "--epochs", "10",
        "--batch_size", "64",
    ]

    print("=== Running Stage 1 (S1) ===")
    subprocess.run(cmd_s1, check=True)

    # ---- IMPORTANT ----
    # Stage 1 must save: "stage1_classic.pth"
    checkpoint = "stage1_classic.pth"
    # -------------------

    # Stage 2 (use checkpoint)
    cmd_s2 = [
        sys.executable,
        "train_dqn_with_tabular_s2.py",
        "--episodes", "10",
        "--epochs", "10",
        "--batch_size", "64",
        "--checkpoint", checkpoint,
    ]

    print("=== Running Stage 2 (S2) with checkpoint ===")
    subprocess.run(cmd_s2, check=True)

    print("=== Finished staged training ===")

if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys
from pathlib import Path


def run_stage(script_name: str, fast: bool = False) -> None:
    """
    Run a training script as a subprocess.
    If --fast is True, pass it down to the child script.
    """
    script_path = Path(script_name)

    if not script_path.exists():
        raise FileNotFoundError(f"Could not find {script_name} in current directory.")

    cmd = [sys.executable, str(script_path)]
    if fast:
        cmd.append("--fast")

    print(f"\n=== Running {script_name} {'(fast mode)' if fast else ''} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"=== Finished {script_name} ===\n")


def main():
    parser = argparse.ArgumentParser(description="Staged DQN training for Pac-Man")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode for both Stage 1 and Stage 2 (e.g. fewer episodes).",
    )
    args = parser.parse_args()

    # Stage 1: classic only (from scratch or tabular init, handled inside s1 script)
    run_stage("train_dqn_with_tabular_s1.py", fast=args.fast)

    # Stage 2: multi-layout training (spiral, spiral_harder, empty), handled by s2 script
    run_stage("train_dqn_with_tabular_s2.py", fast=args.fast)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run 4 ACE translation experiments automatically.

This demonstrates ACE's learning across multiple iterations on the same task.
All metrics are logged to .data/runs.json for analysis.

Usage:
    python run_experiment.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).parent
WORKSPACE_DIR = DEMO_DIR / "workspace"
DATA_DIR = Path(os.getenv("ACE_DEMO_DATA_DIR", str(DEMO_DIR / ".data")))


def run_reset_workspace():
    """Reset workspace (keeps playbook)."""
    print("\n" + "=" * 70)
    print("🔄 Resetting workspace...")
    print("=" * 70 + "\n")

    result = subprocess.run(
        ["./reset_workspace.sh"], cwd=DEMO_DIR, capture_output=True, text=True
    )

    if result.returncode != 0:
        print("❌ Reset failed!")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)


def run_ace_loop(run_number):
    """Run ace_loop.py in AUTO_MODE."""
    print("\n" + "=" * 70)
    print(f"🚀 STARTING RUN {run_number}/4")
    print("=" * 70 + "\n")

    start_time = time.time()

    # Set AUTO_MODE environment variable
    env = os.environ.copy()
    env["AUTO_MODE"] = "true"

    result = subprocess.run(
        ["python", "ace_loop.py"],
        cwd=DEMO_DIR,
        env=env,
        capture_output=False,  # Show output in real-time
        text=True,
    )

    elapsed = time.time() - start_time

    print(f"\n⏱️  Run {run_number} completed in {elapsed/60:.1f} minutes")

    return result.returncode == 0


def main():
    """Run 4 experiments."""
    print("\n" + "=" * 70)
    print("🧪 ACE 4-RUN LEARNING EXPERIMENT")
    print("=" * 70)
    print("\nThis will run ACE 4 times on the same Python→TypeScript task.")
    print("Each run uses the playbook from previous runs.")
    print("All metrics logged to .data/runs.json")
    print("\nStarting experiment...\n")

    total_start = time.time()

    for run_num in range(1, 5):
        # Reset workspace (keeps playbook)
        run_reset_workspace()

        # Run ACE loop
        success = run_ace_loop(run_num)

        if not success:
            print(f"\n❌ Run {run_num} failed - stopping experiment")
            break

        # Brief pause between runs
        if run_num < 4:
            time.sleep(2)

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Metrics saved to: {DATA_DIR / 'runs.json'}")
    print("\nAnalyze results:")
    print("  python analyze_runs.py")


if __name__ == "__main__":
    main()

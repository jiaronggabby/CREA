"""CREA experiment entry point. Author: Jia Rong."""
import sys


def main():
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--config", "configs/experiment_final_v1.yaml"]
    from rpv_experiments.cli import main as run
    run()


if __name__ == "__main__":
    main()

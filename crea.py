"""CREA experiment entry point. Author: Jia Rong."""
import argparse
import sys


SUPPORTED_COMMANDS = (
    "plan", "preflight", "prepare-data", "prepare-controls", "run",
    "final-stats", "final-ablations", "final-standard-reporting",
    "final-shift", "final-efficiency",
)

COMMAND_HELP = {
    "final-stats": "Summarize forecast errors and uncertainty",
    "final-ablations": "Compare allocation and feedback ablations",
    "final-shift": "Evaluate relation-shift responses with frozen forecasters",
    "final-standard-reporting": "Report traffic metrics across all twelve leads",
    "final-efficiency": "Measure aggregation time and storage",
}


def build_parser():
    from rpv_experiments.cli import build_parser as shared_parser

    parser = shared_parser()
    parser.prog = "crea"
    parser.set_defaults(config="configs/experiment_final_v1.yaml")
    # Reuse the shared option definitions, restricting both validation and help.
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name in tuple(action.choices):
                if name not in SUPPORTED_COMMANDS:
                    del action.choices[name]
                else:
                    action.choices[name].prog = f"crea {name}"
            action._choices_actions[:] = [
                choice for choice in action._choices_actions
                if choice.dest in SUPPORTED_COMMANDS
            ]
            for choice in action._choices_actions:
                if choice.dest in COMMAND_HELP:
                    choice.help = COMMAND_HELP[choice.dest]
    return parser


def main():
    argv = sys.argv
    args = build_parser().parse_args(argv[1:])
    from rpv_experiments.cli import main as run

    # Pass the public default (or parsed override) to the unchanged dispatcher.
    sys.argv = [argv[0], "--config", args.config, *argv[1:]]
    try:
        run()
    finally:
        sys.argv = argv


if __name__ == "__main__":
    main()

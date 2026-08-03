"""``verify`` subcommand: registered into the ``aibom`` umbrella CLI.

Only argument parsing and the JSON output contract are implemented so far.
The fingerprinting pipeline itself (ref resolution, architecture-hash gate,
block-0 shape/byte comparison, verdict synthesis) is tracked as individual
FR issues under Milestone 1:
https://github.com/RedHatResearch/aibom-security/milestone/1
"""

from __future__ import annotations

import argparse
import json

from aibom_verifier.types import VerificationResult

NAME = "verify"
HELP = "Verify whether a model's declared base_model claim holds up against its weights"

NOT_IMPLEMENTED_EXIT_CODE = 3


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the ``verify`` subcommand's arguments."""
    parser.add_argument(
        "target", help="Hugging Face repo id of the model being verified, e.g. org/model"
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Claimed base model repo id. Defaults to the target's model card base_model field.",
    )
    parser.add_argument(
        "--revision-target", default=None, help="Pin the target to a specific revision/commit SHA"
    )
    parser.add_argument(
        "--revision-base", default=None, help="Pin the base to a specific revision/commit SHA"
    )
    parser.add_argument("--cache-dir", default=None, help="Override the local cache directory")


def run(args: argparse.Namespace) -> int:
    """Execute the ``verify`` subcommand and print a :class:`VerificationResult` as JSON."""
    result = VerificationResult(
        target=args.target,
        base=args.base,
        verdict="insufficient_evidence",
        message=(
            "The fingerprinting pipeline is not implemented yet — "
            "see Milestone 1: https://github.com/RedHatResearch/aibom-security/milestone/1"
        ),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return NOT_IMPLEMENTED_EXIT_CODE

"""Unified command line for the review evaluation harness."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from review_tuner.compare import main as compare_main
from review_tuner.evaluate import baseline_main
from review_tuner.evaluate import main as evaluate_main
from review_tuner.lint import main as lint_main
from review_tuner.report import report_main, slices_main
from review_tuner.studio import main as studio_main

CommandHandler = Callable[[list[str] | None], int]

COMMANDS: dict[str, CommandHandler] = {
    "eval": evaluate_main,
    "baseline": baseline_main,
    "compare": compare_main,
    "lint": lint_main,
    "slices": slices_main,
    "studio": studio_main,
    "report": report_main,
}

HELP = """usage: review-eval <command> [options]

Score a review bot against a golden set. Torch is not required.

Legacy flags still work: if the first argument starts with '-', the evaluator
runs directly (`review-eval --golden ...`). `review-eval --help` therefore shows
evaluator flags. This message is `review-eval` or `review-eval help`.

commands:
  eval       Score predictions against a golden JSONL
  baseline   Write heuristic baseline predictions
  compare    Diff two prediction files against golden
  lint       Dataset quality checks
  slices     Composite by language, severity, and tag
  studio     Serve the local web studio
  report     Render Markdown/HTML from an eval JSON
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "help":
        print(HELP, end="")
        return 0
    if args[0].startswith("-"):
        return evaluate_main(args)
    command, rest = args[0], args[1:]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"unknown command: {command}", file=sys.stderr)
        print(HELP, end="", file=sys.stderr)
        return 1
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())

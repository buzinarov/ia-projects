"""Run every skill's evaluation and write all artifacts/ summaries.

This is the one command that reproduces the project's reported numbers:

    python -m src.run_all                 # all four skills
    python -m src.run_all --n 1000        # larger triage eval slice

The first run downloads the pre-trained models from the Hugging Face Hub and
the Edmunds dataset; both are cached afterwards. Each skill is independent, so
a failure in one (e.g. a model download) is reported and the rest continue.
"""
import argparse
import json
import traceback

from .evaluate import (
    evaluate_answer,
    evaluate_digest,
    evaluate_translate,
    evaluate_triage,
)


def main(n_triage=500):
    results = {}
    steps = [
        ("triage", lambda: evaluate_triage(n=n_triage)),
        ("translate", evaluate_translate),
        ("answer", evaluate_answer),
        ("digest", evaluate_digest),
    ]
    for name, fn in steps:
        print(f"\n=== Evaluating: {name} ===")
        try:
            results[name] = fn()
        except Exception as exc:  # one skill's failure shouldn't sink the run
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n=== Summary ===")
    print(json.dumps(
        {k: ("error" if "error" in v else "ok") for k, v in results.items()},
        indent=2,
    ))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all skills.")
    parser.add_argument("--n", type=int, default=500, help="triage eval-set cap")
    args = parser.parse_args()
    main(n_triage=args.n)

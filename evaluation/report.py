import argparse
from collections import defaultdict
from pathlib import Path

from evaluation.schema import EvaluationResult, load_results


DEFAULT_RESULTS = Path("evaluation/results.jsonl")
DEFAULT_REPORT = Path("evaluation_results.md")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def scored_values(rows: list[EvaluationResult]) -> list[float]:
    return [row.score for row in rows if row.score is not None]


def compact(text: str, limit: int = 120) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def grouped_by_strategy(results: list[EvaluationResult]) -> dict[str, list[EvaluationResult]]:
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[result.strategy].append(result)
    return dict(grouped)


def render_report(results: list[EvaluationResult]) -> str:
    lines = [
        "# Evaluation Results",
        "",
        "| Strategy | Rows | Avg score | Document hit | Citation recall | MRR | Context precision | Avg latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, rows in sorted(grouped_by_strategy(results).items()):
        lines.append(
            "| "
            f"{strategy} | "
            f"{len(rows)} | "
            f"{average(scored_values(rows)):.3f} | "
            f"{average([row.document_hit for row in rows]):.3f} | "
            f"{average([row.citation_recall for row in rows]):.3f} | "
            f"{average([row.reciprocal_rank for row in rows]):.3f} | "
            f"{average([row.source_precision for row in rows]):.3f} | "
            f"{average([row.latency for row in rows]):.2f}s |"
        )

    lines.extend(
        [
            "",
            "## Low Score Rows",
            "",
            "| ID | Strategy | Score | Source precision | Question | Details |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    scored_results = [result for result in results if result.score is not None]
    for result in sorted(
        scored_results,
        key=lambda row: (row.score, row.source_precision),
    )[:30]:
        lines.append(
            "| "
            f"{result.id} | "
            f"{result.strategy} | "
            f"{result.score:.3f} | "
            f"{result.source_precision:.3f} | "
            f"{compact(result.question)} | "
            f"{compact(result.details, 80)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a markdown report from JSONL evaluation results.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    Path(args.output).write_text(render_report(results), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

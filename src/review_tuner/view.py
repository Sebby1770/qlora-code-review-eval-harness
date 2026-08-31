"""Human-readable eval views shared by the studio and HTML reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from review_tuner.metrics import (
    ExampleScore,
    aggregate_scores,
    error_analysis,
    score_example,
    tokenize,
)
from review_tuner.schema import DatasetError, Prediction, ReviewExample

# Keep in lockstep with score_example weights.
COMPOSITE_WEIGHTS: tuple[tuple[str, float, str], ...] = (
    ("token_f1", 0.25, "Shared words with the expected comment"),
    ("bleu_lite", 0.10, "Short phrase overlap (BLEU-lite)"),
    ("rouge_l", 0.10, "Longest matching word sequence"),
    ("length_ratio", 0.05, "Similar comment length"),
    ("must_mention_recall", 0.20, "Required phrases actually mentioned"),
    ("severity_accuracy", 0.15, "Correct seriousness label"),
    ("tag_f1", 0.10, "Issue tags (security, tests, …)"),
    ("forbidden_rate", 0.05, "Avoided banned phrases"),
)

LETTER_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.80, "A"),
    (0.65, "B"),
    (0.50, "C"),
    (0.35, "D"),
    (0.00, "F"),
)

METRIC_PLAIN: dict[str, str] = {
    "composite": "Overall grade, combining every signal below.",
    "token_f1": "How many words the bot reused from the expected comment.",
    "bleu_lite": "Whether short phrases (1–2 words) match the expected comment.",
    "rouge_l": "Whether the longest stretch of matching words is healthy.",
    "length_ratio": "Whether the bot was much shorter or longer than expected.",
    "must_mention_recall": "Share of required phrases the bot actually said.",
    "forbidden_rate": "Share of banned phrases that slipped in (lower is better).",
    "severity_accuracy": "Whether the bot labelled the issue as serious as the golden set.",
    "tag_f1": "Whether the bot tagged the same issue families (security, tests, …).",
    "exact_match": "Word-for-word match after ignoring punctuation. Rare, and not required.",
}

GLOSSARY: list[dict[str, str]] = [
    {
        "term": "Golden set",
        "plain": (
            "A labelled exam. Each item is a code diff plus the comment a careful "
            "reviewer would write, and the phrases that comment must include."
        ),
    },
    {
        "term": "Prediction",
        "plain": (
            "What the bot (or a human, or the built-in baseline) actually wrote "
            "for that diff."
        ),
    },
    {
        "term": "Required phrases",
        "plain": (
            "The few facts a good comment cannot skip — for example "
            "“expired refresh tokens”. We look for those phrases after ignoring "
            "punctuation and case."
        ),
    },
    {
        "term": "Banned phrases",
        "plain": (
            "Things we do not want, such as nitpicking style when the bug is a "
            "security hole."
        ),
    },
    {
        "term": "Baseline",
        "plain": (
            "A deterministic stand-in that never downloads a model. It guesses "
            "from keywords in the diff so you can try the studio immediately."
        ),
    },
    {
        "term": "Composite",
        "plain": (
            "One number in [0, 1] that mixes overlap, required phrases, severity, "
            "tags, and banned-phrase rate. The letter grade is this number in costume."
        ),
    },
    {
        "term": "Confidence interval",
        "plain": (
            "If we resampled these examples, the overall score would usually land "
            "in this band. Small golden sets make a wide band — that is honesty, "
            "not a bug."
        ),
    },
]


def letter_grade(composite: float) -> str:
    """Map a composite in [0, 1] to a letter grade."""

    score = max(0.0, min(1.0, float(composite)))
    for threshold, letter in LETTER_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


def verdict(composite: float, threshold: float = 0.60) -> str:
    """Pass / Weak / Fail against a ship-gate threshold."""

    score = float(composite)
    if score >= threshold:
        return "Pass"
    if score >= threshold * 0.75:
        return "Weak"
    return "Fail"


def percent(value: float) -> str:
    return f"{max(0.0, min(1.0, float(value))) * 100:.0f}%"


def explain_score(score: ExampleScore) -> list[dict[str, Any]]:
    """Waterfall of composite terms for one example."""

    rows: list[dict[str, Any]] = []
    for key, weight, plain in COMPOSITE_WEIGHTS:
        raw = float(getattr(score, key))
        value = (1.0 - raw) if key == "forbidden_rate" else raw
        contribution = weight * value
        rows.append(
            {
                "key": key,
                "weight": weight,
                "value": round(value, 6),
                "raw": round(raw, 6),
                "contribution": round(contribution, 6),
                "plain": plain,
            }
        )
    return rows


def _example_verdict(score: ExampleScore) -> str:
    if score.composite >= 0.75 and score.must_mention_recall >= 0.99:
        return "caught"
    if score.must_mention_recall < 0.5 or score.forbidden_rate > 0:
        return "missed"
    return "partial"


def overlapping_tokens(prediction: str, expected: str, limit: int = 24) -> list[str]:
    """Shared tokens in first-seen order from the prediction."""

    gold = set(tokenize(expected))
    seen: set[str] = set()
    shared: list[str] = []
    for token in tokenize(prediction):
        if token not in gold or token in seen:
            continue
        seen.add(token)
        shared.append(token)
        if len(shared) >= limit:
            break
    return shared


def grade_histogram(examples: list[dict[str, Any]]) -> dict[str, int]:
    hist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for row in examples:
        letter = str(row.get("grade") or letter_grade(float(row.get("composite") or 0.0)))
        if letter in hist:
            hist[letter] += 1
    return hist


def tag_breakdown(examples: list[dict[str, Any]]) -> dict[str, float]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for row in examples:
        tags = [str(tag) for tag in (row.get("tags") or []) if str(tag).strip()]
        composite = float(row.get("composite") or 0.0)
        if not tags:
            bucket["(untagged)"].append(composite)
            continue
        for tag in tags:
            bucket[tag].append(composite)
    return {tag: round(sum(vals) / len(vals), 6) for tag, vals in sorted(bucket.items())}


def decorate_example(
    golden: ReviewExample,
    prediction: Prediction,
    score: ExampleScore,
) -> dict[str, Any]:
    """Full inspector payload for one golden/prediction pair."""

    hits = [
        phrase
        for phrase in golden.must_mention
        if phrase not in score.missed_must_mention
    ]
    payload = score.as_dict()
    payload.update(
        {
            "file_path": golden.file_path,
            "context": golden.context,
            "diff": golden.diff,
            "expected_comment": golden.target_comment,
            "prediction": prediction.prediction,
            "tags": list(golden.tags),
            "predicted_tags": list(prediction.tags),
            "must_mention": list(golden.must_mention),
            "avoid": list(golden.avoid),
            "must_mention_hits": hits,
            "grade": letter_grade(score.composite),
            "verdict": _example_verdict(score),
            "explanation": explain_score(score),
            "overlap_tokens": overlapping_tokens(
                prediction.prediction, golden.target_comment
            ),
        }
    )
    return payload


def _security_miss_line(examples: list[dict[str, Any]]) -> str | None:
    security = [
        row
        for row in examples
        if "security" in {tag.lower() for tag in row.get("tags") or []}
    ]
    if not security:
        return None
    missed = [row for row in security if float(row.get("must_mention_recall") or 0) < 1.0]
    if not missed:
        n = len(security)
        return f"Security cases: the bot mentioned every required phrase on {n}/{n}."
    return (
        f"Security cases: required phrases were incomplete on {len(missed)} of {len(security)}."
    )


def build_story(
    aggregate: dict[str, Any],
    analysis: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    threshold: float = 0.60,
) -> str:
    """One-paragraph English summary of a run."""

    count = int(aggregate.get("count") or 0)
    composite = float(aggregate.get("composite") or 0.0)
    letter = letter_grade(composite)
    gate = verdict(composite, threshold)
    mention = float(aggregate.get("must_mention_recall") or 0.0)
    parts = [
        f"Overall {gate.lower()} — letter {letter} ({percent(composite)} across {count} example"
        f"{'' if count == 1 else 's'})."
    ]
    missed = analysis.get("most_missed_must_mention") or []
    if missed:
        top = missed[0]
        phrase = str(top.get("phrase") or "").strip()
        n = int(top.get("count") or 0)
        if phrase:
            parts.append(
                f"The most-skipped required phrase was “{phrase}” ({n} example"
                f"{'' if n == 1 else 's'})."
            )
    else:
        parts.append("Every required phrase appeared at least once.")
    security = _security_miss_line(examples)
    if security:
        parts.append(security)
    parts.append(f"Required-phrase coverage averaged {percent(mention)}.")
    ci_lo = aggregate.get("composite_ci_lo")
    ci_hi = aggregate.get("composite_ci_hi")
    if ci_lo is not None and ci_hi is not None:
        parts.append(
            f"We are 95% sure the true overall score sits between {percent(float(ci_lo))} "
            f"and {percent(float(ci_hi))}."
        )
    return " ".join(parts)


def language_severity_heatmap(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean composite by language × gold severity for the studio grid."""

    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    languages: list[str] = []
    severities: list[str] = []
    for row in examples:
        lang = str(row.get("language") or "unknown")
        sev = str(row.get("severity") or "unknown")
        if lang not in languages:
            languages.append(lang)
        if sev not in severities:
            severities.append(sev)
        bucket[(lang, sev)].append(float(row.get("composite") or 0.0))

    cells = []
    for lang in languages:
        for sev in severities:
            values = bucket.get((lang, sev))
            if not values:
                continue
            cells.append(
                {
                    "language": lang,
                    "severity": sev,
                    "count": len(values),
                    "composite": round(sum(values) / len(values), 6),
                }
            )
    return {"languages": languages, "severities": severities, "cells": cells}


def compare_views(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Metric deltas plus examples that flipped caught/missed."""

    keys = [
        "composite",
        "token_f1",
        "bleu_lite",
        "rouge_l",
        "must_mention_recall",
        "severity_accuracy",
        "tag_f1",
        "forbidden_rate",
    ]
    left_agg = left.get("aggregate") or {}
    right_agg = right.get("aggregate") or {}
    delta = {
        key: round(float(right_agg.get(key, 0.0)) - float(left_agg.get(key, 0.0)), 6)
        for key in keys
        if key in left_agg and key in right_agg
    }
    left_by_id = {row["id"]: row for row in left.get("examples") or [] if "id" in row}
    right_by_id = {row["id"]: row for row in right.get("examples") or [] if "id" in row}
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    for example_id, left_row in left_by_id.items():
        right_row = right_by_id.get(example_id)
        if right_row is None:
            continue
        left_v = str(left_row.get("verdict"))
        right_v = str(right_row.get("verdict"))
        entry = {
            "id": example_id,
            "from": left_v,
            "to": right_v,
            "left_composite": left_row.get("composite"),
            "right_composite": right_row.get("composite"),
        }
        if left_v != "caught" and right_v == "caught":
            improved.append(entry)
        elif left_v == "caught" and right_v != "caught":
            regressed.append(entry)
    return {
        "delta_b_minus_a": delta,
        "improved": improved,
        "regressed": regressed,
        "a": {"letter_grade": left.get("letter_grade"), "aggregate": left_agg},
        "b": {"letter_grade": right.get("letter_grade"), "aggregate": right_agg},
        "count": int(left_agg.get("count") or 0),
    }


def build_eval_view(
    golden: list[ReviewExample],
    predictions: list[Prediction],
    *,
    threshold: float = 0.60,
    extra_prediction_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Score predictions and return the studio/report view model."""

    if not golden:
        raise DatasetError("no golden examples to score")

    by_id = {item.id: item for item in predictions}
    missing = [example.id for example in golden if example.id not in by_id]
    if missing:
        raise DatasetError(f"missing predictions for ids: {', '.join(missing)}")

    scores: list[ExampleScore] = []
    examples: list[dict[str, Any]] = []
    for example in golden:
        prediction = by_id[example.id]
        score = score_example(example, prediction)
        scores.append(score)
        examples.append(decorate_example(example, prediction, score))

    aggregate = aggregate_scores(scores)
    analysis = error_analysis(scores, golden)
    composite = float(aggregate.get("composite") or 0.0)
    ranked = sorted(examples, key=lambda row: float(row.get("composite") or 0.0))
    worst = ranked[0] if ranked else None
    return {
        "aggregate": aggregate,
        "error_analysis": analysis,
        "examples": examples,
        "letter_grade": letter_grade(composite),
        "verdict": verdict(composite, threshold),
        "threshold": threshold,
        "story": build_story(aggregate, analysis, examples, threshold=threshold),
        "heatmap": language_severity_heatmap(examples),
        "worst_id": None if worst is None else worst["id"],
        "histogram": grade_histogram(examples),
        "by_tag": tag_breakdown(examples),
        "extra_prediction_ids": list(extra_prediction_ids or []),
        "weights": [
            {"key": key, "weight": weight, "plain": plain}
            for key, weight, plain in COMPOSITE_WEIGHTS
        ],
        "glossary": GLOSSARY,
        "metric_plain": METRIC_PLAIN,
        "badge": {
            "letter": letter_grade(composite),
            "verdict": verdict(composite, threshold),
            "percent": round(composite * 100),
        },
    }


def summarize_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage map for the dataset studio (language / severity / tag counts)."""

    languages: Counter[str] = Counter()
    severities: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    for raw in records:
        languages[str(raw.get("language") or "unknown").strip().lower() or "unknown"] += 1
        if "severity" in raw and str(raw.get("severity") or "").strip():
            severities[str(raw.get("severity")).strip().lower()] += 1
        else:
            severities["(missing)"] += 1
        for tag in raw.get("tags") or []:
            label = str(tag).strip().lower()
            if label:
                tags[label] += 1
    return {
        "count": len(records),
        "languages": dict(sorted(languages.items())),
        "severities": dict(sorted(severities.items())),
        "tags": dict(sorted(tags.items(), key=lambda item: (-item[1], item[0]))),
    }


def records_to_examples(records: list[dict[str, Any]]) -> list[ReviewExample]:
    return [
        ReviewExample.from_dict(record, row_number=index)
        for index, record in enumerate(records, start=1)
    ]


def records_to_predictions(records: list[dict[str, Any]]) -> list[Prediction]:
    return [
        Prediction.from_dict(record, row_number=index)
        for index, record in enumerate(records, start=1)
    ]


def unmatched_prediction_ids(
    golden: list[ReviewExample], predictions: list[Prediction]
) -> list[str]:
    known = {example.id for example in golden}
    extra: list[str] = []
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.id in known or prediction.id in seen:
            continue
        seen.add(prediction.id)
        extra.append(prediction.id)
    return extra

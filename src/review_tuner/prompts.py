"""Prompt construction for supervised fine-tuning and inference."""

from __future__ import annotations

from review_tuner.schema import ReviewExample

SYSTEM_PROMPT = (
    "You are a senior software engineer reviewing a pull request. "
    "Write one concise, actionable review comment. Focus on correctness, security, "
    "maintainability, and missing tests. Do not invent files or APIs."
)


def build_prompt(example: ReviewExample) -> str:
    """Build the instruction prompt without the target answer."""

    tags = ", ".join(example.tags) if example.tags else "none provided"
    return "\n".join(
        [
            "### System",
            SYSTEM_PROMPT,
            "",
            "### Pull Request Context",
            f"File: {example.file_path}",
            f"Language: {example.language}",
            f"Known severity: {example.severity}",
            f"Tags: {tags}",
            f"Context: {example.context}",
            "",
            "### Diff",
            "```diff",
            example.diff.strip(),
            "```",
            "",
            "### Task",
            "Return a review comment that identifies the issue, explains the risk, "
            "and suggests a concrete fix. Keep it under 90 words.",
            "",
            "### Review Comment",
        ]
    )


def build_sft_text(example: ReviewExample) -> str:
    """Build a single text sample for causal language model SFT."""

    return f"{build_prompt(example)}\n{example.target_comment.strip()}"


def examples_to_sft_rows(examples: list[ReviewExample]) -> list[dict[str, str]]:
    """Convert validated examples into the text field expected by SFTTrainer."""

    return [{"id": example.id, "text": build_sft_text(example)} for example in examples]

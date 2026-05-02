from review_tuner.prompts import build_prompt, build_sft_text
from review_tuner.schema import ReviewExample


def test_prompt_omits_target_comment() -> None:
    example = ReviewExample(
        id="one",
        diff="- old\n+ new",
        file_path="app.py",
        language="python",
        context="A risky change.",
        target_comment="Please add the missing validation.",
        severity="high",
        tags=("security",),
    )

    prompt = build_prompt(example)

    assert "A risky change." in prompt
    assert "```diff" in prompt
    assert "Please add the missing validation." not in prompt


def test_sft_text_appends_target_comment() -> None:
    example = ReviewExample(
        id="one",
        diff="- old\n+ new",
        file_path="app.py",
        language="python",
        context="A risky change.",
        target_comment="Please add the missing validation.",
        severity="high",
    )

    assert build_sft_text(example).endswith("Please add the missing validation.")

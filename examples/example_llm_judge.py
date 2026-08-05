"""
example_llm_judge.py — use one LLM to judge the output of another.

For demo purposes, the "judge" is a deterministic stub: it scores outputs
based on length + presence of expected keywords.

Run with:  python examples/example_llm_judge.py
"""

from tiny_evaluator import (
    EvaluationSuite,
    TestCase,
    generate_html_report,
)


# The agent being evaluated.
def summarizer_agent(prompt: str) -> str:
    if "climate" in prompt.lower():
        # Bad summary: too short, misses key terms.
        return "It's about climate."
    if "election" in prompt.lower():
        # Good summary: more substantive.
        return (
            "The 2024 election saw record voter turnout, with shifts in "
            "swing states driven largely by economic concerns and climate policy."
        )
    return "Summary unavailable."


# The judge LLM (stubbed). Returns a score in [0, 1].
def judge_llm(prompt: str) -> str:
    # Inspect the "Actual output" section crudely.
    if "Actual output:" in prompt:
        output = prompt.split("Actual output:")[-1].strip()
    else:
        output = prompt
    score = 0.0
    if len(output) > 40:
        score += 0.4
    if len(output) > 80:
        score += 0.3
    if any(w in output.lower() for w in ["turnout", "economy", "policy", "climate"]):
        score += 0.3
    return f"{score:.2f}"


def main() -> None:
    suite = EvaluationSuite(
        name="summarization-judged",
        capabilities=["summarization"],
        test_cases=[
            TestCase(
                id="climate",
                prompt="Summarize the article on climate change.",
                expected_output="(factual, multi-sentence summary)",
                validator="llm_judge",
                metadata={
                    "capability": "summarization",
                    "judge_llm": judge_llm,
                    "judge_threshold": 0.6,
                    "rubric": (
                        "Score the summary 0–1 based on length, factual accuracy, "
                        "and inclusion of climate-related terms."
                    ),
                },
            ),
            TestCase(
                id="election",
                prompt="Summarize the 2024 election results.",
                expected_output="(factual, multi-sentence summary)",
                validator="llm_judge",
                metadata={
                    "capability": "summarization",
                    "judge_llm": judge_llm,
                    "judge_threshold": 0.6,
                    "rubric": (
                        "Score the summary 0–1 based on length, factual accuracy, "
                        "and inclusion of election-related terms."
                    ),
                },
            ),
        ],
    )

    report = suite.run(summarizer_agent)
    print(f"Pass rate: {report.pass_rate:.0%}  ({report.passed}/{report.total})")
    for r in report.results:
        marker = "✓" if r.passed else "✗"
        print(f"  {marker} {r.test_id}: score={r.score:.2f}")

    generate_html_report(report, output_path="examples/_judge_report.html")
    print("Wrote examples/_judge_report.html")


if __name__ == "__main__":
    main()
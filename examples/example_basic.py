"""
example_basic.py — the simplest possible tiny-evaluator use case.

Evaluate a plain Python function against a handful of test cases.
Run with:  python examples/example_basic.py
"""

from tiny_evaluator import (
    EvaluationSuite,
    TestCase,
    generate_html_report,
)


# The "thing under test" — anything callable works.
def add(a: int, b: int) -> int:
    # Oops: this implementation has a bug for negatives.
    return a + b


def make_agent():
    """Wrap `add` so it accepts a string prompt like 'add 1 2'."""
    def agent(prompt: str) -> str:
        # very crude "parser"
        tokens = prompt.lower().replace("add", "").split()
        nums = [int(t) for t in tokens if t.lstrip("-").isdigit()]
        return str(add(nums[0], nums[1]))
    return agent


def main() -> None:
    suite = EvaluationSuite(
        name="basic-arithmetic",
        test_cases=[
            TestCase(id="pos-pos", prompt="add 2 3", expected_output="5"),
            TestCase(id="pos-zero", prompt="add 7 0", expected_output="7"),
            TestCase(id="neg-pos", prompt="add -1 4", expected_output="3"),
            TestCase(id="zero-zero", prompt="add 0 0", expected_output="0"),
        ],
    )

    report = suite.run(make_agent())
    print(f"Pass rate: {report.pass_rate:.0%}  ({report.passed}/{report.total})")
    for r in report.results:
        marker = "✓" if r.passed else "✗"
        print(f"  {marker} {r.test_id}: output={r.output!r}, expected={r.expected!r}")

    generate_html_report(report, output_path="examples/_basic_report.html")
    print("\nWrote examples/_basic_report.html")


if __name__ == "__main__":
    main()
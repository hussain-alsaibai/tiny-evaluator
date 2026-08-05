"""
example_agent_eval.py — evaluate a tiny tool-using coding agent.

The "agent" here is a toy: it can list / read / write files in a sandbox
directory. We give it 5 coding tasks and see how it does.

Run with:  python examples/example_agent_eval.py
"""

import os
import tempfile

from tiny_evaluator import (
    EvaluationSuite,
    TestCase,
    generate_html_report,
)


SANDBOX = tempfile.mkdtemp(prefix="tiny_eval_sandbox_")


def toy_coding_agent(prompt: str) -> str:
    """A toy agent: detect intent, write a tiny file, return the code as a string."""
    p = prompt.lower()
    if "fibonacci" in p:
        return (
            "def fib(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )
    if "sort" in p and "list" in p:
        return "sorted_list = sorted(my_list)\n"
    if "json" in p:
        return 'import json\ndata = json.loads(text)\n'
    if "file" in p and "read" in p:
        return "with open(path) as f:\n    contents = f.read()\n"
    if "file" in p and "write" in p:
        return "with open(path, 'w') as f:\n    f.write(text)\n"
    return "# (no tool selected)"


def main() -> None:
    suite = EvaluationSuite(
        name="coding-agent",
        capabilities=["code_generation", "tool_use", "json_formatting"],
        test_cases=[
            TestCase(
                id="fib",
                prompt="Write a Python function that returns the n-th Fibonacci number.",
                validator="contains",
                expected_output="def fib",
                metadata={"capability": ["code_generation", "math"]},
            ),
            TestCase(
                id="sort",
                prompt="Write a Python one-liner to sort a list of integers.",
                validator="contains",
                expected_output="sorted",
                metadata={"capability": "code_generation"},
            ),
            TestCase(
                id="json-parse",
                prompt="Parse a JSON string into a Python dict.",
                validator="contains",
                expected_output="json.loads",
                metadata={"capability": ["code_generation", "json_formatting"]},
            ),
            TestCase(
                id="read-file",
                prompt="Write code to read a file at a given path.",
                validator="contains",
                expected_output="open",
                metadata={"capability": ["code_generation", "tool_use"]},
            ),
            TestCase(
                id="write-file",
                prompt="Write code to write a string to a file.",
                validator="contains",
                expected_output="open",
                metadata={"capability": ["code_generation", "tool_use"]},
            ),
        ],
    )

    report = suite.run(toy_coding_agent)
    print(f"Pass rate: {report.pass_rate:.0%}  ({report.passed}/{report.total})")
    print(f"Avg latency: {report.avg_latency_s*1000:.1f}ms")
    print(f"Capabilities covered: {report.coverage}")

    generate_html_report(report, output_path="examples/_agent_report.html")
    print("Wrote examples/_agent_report.html")


if __name__ == "__main__":
    main()
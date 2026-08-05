"""
example_compare_llms.py — compare three "LLMs" on the same suite.

In real life, swap the agent callables for `openai.ChatCompletion.create`,
`anthropic.Anthropic().messages.create`, and your local Ollama HTTP call.

Run with:  python examples/example_compare_llms.py
"""

from tiny_evaluator import (
    Benchmark,
    EvaluationSuite,
    TestCase,
    diff_benchmark,
    generate_html_report,
)


# Three fake "LLMs" of varying quality.
def openai_like(prompt: str) -> str:
    return "Paris" if "france" in prompt.lower() else "4"


def anthropic_like(prompt: str) -> str:
    return "Paris" if "france" in prompt.lower() else "5"


def ollama_like(prompt: str) -> str:
    return "London" if "france" in prompt.lower() else "4"


def main() -> None:
    suite = EvaluationSuite(
        name="trivial",
        test_cases=[
            TestCase(id="geo",  prompt="Capital of France?", expected_output="Paris"),
            TestCase(id="math", prompt="2+2?",               expected_output="4"),
            TestCase(id="math2", prompt="3+2?",              expected_output="5"),
        ],
    )

    bench = Benchmark("openai-vs-anthropic-vs-ollama", suites=[suite])
    bench.add_run("openai",    openai_like)
    bench.add_run("anthropic", anthropic_like)
    bench.add_run("ollama",    ollama_like)
    bench.set_baseline("openai")

    report = bench.run()

    print(f"{'label':<12} {'suite':<10} {'pass':<6} {'avg_score':<10} {'avg_latency_s':<12}")
    for e in report.entries:
        print(f"{e.label:<12} {e.suite_name:<10} "
              f"{e.report.passed}/{e.report.total:<5} "
              f"{e.report.avg_score:<10.2f} "
              f"{e.report.avg_latency_s:<12.4f}")

    diff = diff_benchmark(report, baseline_label="openai")
    print("\nDeltas vs openai:")
    for suite_name, by_label in diff.items():
        for label, vals in by_label.items():
            d = vals["delta_vs_baseline"]
            print(f"  {label:<10} pass_rate Δ={d['pass_rate']:+.2f} "
                  f"score Δ={d['avg_score']:+.2f} latency Δ={d['avg_latency_s']:+.3f}s")

    generate_html_report(report, output_path="examples/_compare_report.html")
    print("Wrote examples/_compare_report.html")


if __name__ == "__main__":
    main()
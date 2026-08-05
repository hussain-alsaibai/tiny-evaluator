# 🧪 tiny-evaluator

> Zero-dependency LLM evaluation harness for AI agents and tool-using systems.
> Like HELM or BigBench, but for your own agentic code.

[![version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/hussain-alsaibai/tiny-evaluator/releases)
![python](https://img.shields.io/badge/python-≥3.8-green)
[![license](https://img.shields.io/badge/license-MIT-purple)](./LICENSE)
![deps](https://img.shields.io/badge/deps-zero-brightgreen)

```text
pip install tiny-evaluator
```

---

## What is it?

**tiny-evaluator** is a tiny, dependency-free Python library that lets you
benchmark, regression-test, and compare LLMs and AI agents — including
tool-using and streaming ones. It's designed to feel like a unit-test runner,
not a research framework.

- **Zero deps.** Pure Python stdlib. No `pip install` gymnastics.
- **Agent-friendly.** Pass any callable `prompt → output` as the thing-under-test.
- **Streaming-aware.** Capture per-token latency for streaming models.
- **LLM-as-judge.** Use one LLM to grade another LLM's outputs.
- **Coverage tracking.** Know which capabilities (reasoning, tool-use, math, …)
  your suite actually exercises.
- **Self-contained HTML reports.** Dark-themed, single-file, no JS required.

It won't replace HELM for leaderboard runs, but it **will** let you ship a
regression suite for your agent in 5 minutes.

---

## ✨ Features

| Feature | What it does |
|--------|--------------|
| `EvaluationSuite` | A named bag of `TestCase`s with metadata + coverage |
| `Evaluator` | Runs a `TestCase` against any agent; returns `TestResult` |
| `StreamingEvaluator` | Captures tokens, time-to-first-token, tokens/sec |
| `Benchmark` | Run the same suites against multiple agents, get a comparison |
| `CoverageTracker` | Track which capabilities (math, reasoning, RAG, …) you tested |
| `equals`, `contains`, `regex_match`, `json_schema_valid` | Built-in deterministic validators |
| `llm_judge` | LLM-as-judge with rubric + threshold |
| `generate_html_report` | Self-contained dark-mode HTML report |

---

## 🚀 Quick Start

```python
from tiny_evaluator import (
    EvaluationSuite, TestCase, Evaluator, generate_html_report,
)

# 1. Define a tiny "agent" — anything callable.
def my_agent(prompt: str) -> str:
    if "capital of France" in prompt:
        return "Paris"
    if "2+2" in prompt:
        return "4"
    return "I don't know"

# 2. Build a test suite.
suite = EvaluationSuite(
    name="smoke",
    test_cases=[
        TestCase(id="geo",  prompt="What is the capital of France?", expected_output="Paris"),
        TestCase(id="math", prompt="What is 2+2?",                  expected_output="4"),
        TestCase(id="miss", prompt="Who won the 1962 World Cup?",  expected_output="Brazil"),
    ],
)

# 3. Run it.
report = suite.run(my_agent)
print(f"Pass rate: {report.pass_rate:.0%}")   # 67%

# 4. Render an HTML report.
generate_html_report(report, output_path="report.html")
```

Output:

```
Pass rate: 67%
```

---

## 🧑‍💻 Examples

### 1. Evaluate a tiny AI coding agent on 5 tasks

See [`examples/example_agent_eval.py`](./examples/example_agent_eval.py).

```python
from tiny_evaluator import EvaluationSuite, TestCase

suite = EvaluationSuite(
    name="coding",
    test_cases=[
        TestCase(id="fib",  prompt="Write a Python function that returns the n-th Fibonacci number.",
                 expected_output="def fib(n):", validator="contains"),
        TestCase(id="sort", prompt="Write a Python one-liner to sort a list of integers.",
                 expected_output="sorted",    validator="contains"),
        TestCase(id="json", prompt='Return JSON: {"name": "Alice", "age": 30}',
                 expected_output=None,       validator="json_schema_valid",
                 metadata={"schema": {"type": "object", "required": ["name", "age"]}}),
    ],
)
```

### 2. Compare two LLMs on the same test suite

See [`examples/example_compare_llms.py`](./examples/example_compare_llms.py).

```python
from tiny_evaluator import Benchmark, EvaluationSuite, TestCase

suite = EvaluationSuite(name="truth", test_cases=[
    TestCase(id="t1", prompt="hi",  expected_output="hello"),
    TestCase(id="t2", prompt="bye", expected_output="goodbye"),
])

bench = Benchmark("gpt-vs-claude", suites=[suite])
bench.add_run("gpt-4",    lambda p: call_openai(p))
bench.add_run("claude-3", lambda p: call_anthropic(p))

report = bench.run()
# report.entries → 2 entries per suite
# diff_benchmark(report, baseline_label="gpt-4") → deltas
```

### 3. Use LLM-as-judge for subjective evaluation

See [`examples/example_llm_judge.py`](./examples/example_llm_judge.py).

```python
TestCase(
    id="summary-quality",
    prompt="Summarize the following article: ...",
    expected_output="(see article)",
    validator="llm_judge",
    metadata={
        "judge_llm":  lambda p: my_other_llm(p),
        "rubric":     "Score 0–1 for factual accuracy and conciseness.",
        "judge_threshold": 0.7,
    },
)
```

### 4. Basic functional test

See [`examples/example_basic.py`](./examples/example_basic.py). The simplest
possible case: a function, a few assertions.

---

## 📊 HTML Report

`generate_html_report(report_or_results, output_path="report.html")` writes a
single-file, dark-mode HTML report containing:

- **Summary cards** — passed / failed / errors / avg score / avg & p95 latency
- **Pass-rate bar** — CSS-only progress bar
- **Capability coverage heatmap** — grid showing each capability's coverage status
- **Results table** — every test with status, score, latency, and capability tag
- **Expandable failure details** — `<details>` blocks with prompt, expected, actual, error

Open `report.html` in any browser — no server, no JS bundler.

---

## 🔍 Comparison

| Feature | tiny-evaluator | LangChain evals | RAGAS | HELM |
|---------|:--------------:|:---------------:|:-----:|:----:|
| Zero dependencies | ✅ | ❌ | ❌ | ❌ |
| Python stdlib only | ✅ | ❌ | ❌ | ❌ |
| Tool-use / agent eval | ✅ | partial | partial | ❌ |
| Streaming latency | ✅ | ❌ | ❌ | ❌ |
| Capability coverage | ✅ | ❌ | ❌ | partial |
| Self-contained HTML report | ✅ | ❌ | partial | ❌ |
| LLM-as-judge built-in | ✅ | partial | ✅ | ✅ |
| Side-by-side model compare | ✅ | partial | ❌ | ✅ |
| Setup time | **minutes** | hours | hours | days |

tiny-evaluator is **not** a leaderboard framework. If you need thousands of
prompts, statistical significance tests, and multi-thousand-dollar GPU runs,
use HELM. tiny-evaluator is for the 90% case: "did my agent get worse today?"

---

## 📚 API Reference

```python
@dataclass
class TestCase:
    id: str
    prompt: str
    expected_output: Any = None
    validator: Union[Validator, str, None] = None
    metadata: Dict[str, Any] = {}

@dataclass
class TestResult:
    test_id: str
    passed: bool
    score: float                       # 0..1
    latency_s: float
    output: Any = None
    expected: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class EvaluationSuite:
    def __init__(name, test_cases=[], capabilities=None)
    def add(test_case) / extend(test_cases)
    def run(agent)                -> SuiteReport
    def run_streaming(streaming_agent) -> SuiteReport

class Evaluator:
    def __init__(agent)
    def run(test_case) -> TestResult

class StreamingEvaluator:
    def __init__(streaming_agent, on_token=None)
    def run(test_case) -> TestResult

class Benchmark:
    def __init__(name, suites)
    def add_run(label, agent)
    def set_baseline(label)
    def run() -> BenchmarkReport

class CoverageTracker:
    def record(test_case) / record_suite(suite)
    def coverage_ratio() -> float
    def heatmap() -> Dict[capability, (count, status)]

# Built-in validators (factory functions)
equals(expected)
contains(substr_or_sequence)
regex_match(pattern, flags=0)
json_schema_valid(schema=None)
llm_judge(judge_llm, rubric=..., threshold=0.5)

# Reports
generate_html_report(suite_report_or_results, output_path=None) -> str
diff_benchmark(benchmark_report, baseline_label=None) -> dict
```

---

## 🛣️ Roadmap (v0.2)

- [ ] Pluggable result sinks (CSV, JSONL, SQLite)
- [ ] Async evaluator (`asyncio` based, no external deps)
- [ ] Statistical significance helpers (paired bootstrap, McNemar)
- [ ] Built-in deterministic "judge" heuristics (length, no-refusal, JSON well-formed)
- [ ] Cost tracker (token-count → USD) via metadata
- [ ] Suite versioning + diff against a stored baseline JSON
- [ ] Tiny web UI (stdlib `http.server`) for browsing reports

Have a feature request? [Open an issue](https://github.com/hussain-alsaibai/tiny-evaluator/issues).

---

## 🧪 Running the tests

```bash
pip install pytest        # only needed for the `pytest` runner
pytest test_tiny_evaluator.py
# or, no install:
python -m unittest test_tiny_evaluator.py
```

---

## 📄 License

[MIT](./LICENSE) © 2026 Hussain Al-Saibai
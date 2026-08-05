"""
tiny_evaluator.py

Zero-dependency LLM evaluation harness for AI agents, LLMs, and tool-using systems.

Like HELM or BigBench, but for your own agentic code.

Public API:
    - TestCase, TestResult dataclasses
    - EvaluationSuite, Evaluator, Benchmark classes
    - StreamingEvaluator, CoverageTracker
    - Validators: equals, contains, regex_match, json_schema_valid, llm_judge
    - generate_html_report(results)
"""

from __future__ import annotations

import dataclasses
import functools
import html as _html
import json
import re
import statistics
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EvaluatorError(Exception):
    """Base exception for tiny_evaluator errors."""


class ValidatorError(EvaluatorError):
    """Raised when a validator is malformed or fails unexpectedly."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

Validator = Callable[["TestResult"], bool]


@dataclass
class TestCase:
    """A single evaluation scenario.

    Attributes:
        id:        unique identifier (used in reports, deduplication, lookups)
        prompt:    the input sent to the agent/LLM under test
        expected_output: optional ground truth (free-form or structured)
        validator: a callable that takes a TestResult and returns bool,
                   or one of the strings "equals"/"contains"/"regex_match"/
                   "json_schema_valid"/"llm_judge" for built-ins
        metadata:  free-form dict for tags, capability hints, dataset info, etc.
    """
    id: str
    prompt: str
    expected_output: Any = None
    validator: Union[Validator, str, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """The result of running a single TestCase against an agent/LLM."""
    test_id: str
    passed: bool
    score: float                       # 0.0–1.0
    latency_s: float                   # wall clock seconds
    output: Any = None
    expected: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------

def equals(expected: Any) -> Validator:
    """Return a validator that compares output to expected with ==."""
    def _v(r: TestResult) -> bool:
        return r.output == expected
    _v.__name__ = f"equals({expected!r})"
    return _v


def contains(substr: Union[str, Sequence[Any]]) -> Validator:
    """Return a validator that checks whether the output contains `substr`."""
    if isinstance(substr, str):
        def _v(r: TestResult) -> bool:
            return isinstance(r.output, str) and substr in r.output
        _v.__name__ = f"contains({substr!r})"
        return _v
    # Sequence of allowed substrings/elements (all must appear)
    parts = list(substr)
    def _v(r: TestResult) -> bool:
        return all(p in r.output for p in parts)
    _v.__name__ = f"contains-all({parts!r})"
    return _v


def regex_match(pattern: str, flags: int = 0) -> Validator:
    """Return a validator that matches the output against a regex pattern."""
    rx = re.compile(pattern, flags)
    def _v(r: TestResult) -> bool:
        return isinstance(r.output, str) and bool(rx.search(r.output))
    _v.__name__ = f"regex_match({pattern!r})"
    return _v


def json_schema_valid(schema: Optional[Dict[str, Any]] = None) -> Validator:
    """Return a validator that checks output is valid JSON, optionally against a schema.

    With no schema: just verifies the output parses as JSON.
    With a schema: performs a lightweight structural validation:
      - checks required keys are present
      - checks type hints ("type": "string"|"number"|"object"|"array"|"boolean"|"null")
      - checks enum membership if "enum" is provided
    """
    def _v(r: TestResult) -> bool:
        try:
            data = json.loads(r.output) if isinstance(r.output, str) else r.output
        except (ValueError, TypeError):
            return False
        if schema is None:
            return True
        return _validate_schema(data, schema)
    _v.__name__ = "json_schema_valid"
    return _v


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": (int,),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _validate_schema(data: Any, schema: Dict[str, Any]) -> bool:
    """Minimal JSON-schema-style structural validator (no external deps)."""
    if "type" in schema:
        t = schema["type"]
        py = _TYPE_MAP.get(t)
        if py is None:
            raise ValidatorError(f"Unknown schema type: {t!r}")
        # bool is a subclass of int; handle carefully
        if t == "boolean":
            if not isinstance(data, bool):
                return False
        elif not isinstance(data, py):
            return False
    if "enum" in schema and data not in schema["enum"]:
        return False
    if "required" in schema and isinstance(data, dict):
        for k in schema["required"]:
            if k not in data:
                return False
    if isinstance(data, dict):
        for k, v in data.items():
            if k in schema.get("properties", {}):
                if not _validate_schema(v, schema["properties"][k]):
                    return False
    if isinstance(data, list) and "items" in schema:
        for item in data:
            if not _validate_schema(item, schema["items"]):
                return False
    return True


def llm_judge(judge_llm: Callable[[str], str],
              rubric: str = "Rate the response on a 0–1 scale. Reply with only a number.",
              threshold: float = 0.5) -> Validator:
    """Return a validator that asks an LLM to judge the output.

    Args:
        judge_llm: callable taking a prompt string and returning a string
        rubric:    prompt template sent to the judge
        threshold: minimum score to count as pass
    """
    def _v(r: TestResult) -> bool:
        prompt = (
            f"{rubric}\n\n"
            f"--- Test case ---\n{r.test_id}\n\n"
            f"--- Expected ---\n{r.expected}\n\n"
            f"--- Actual output ---\n{r.output}"
        )
        raw = judge_llm(prompt).strip()
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not m:
            return False
        score = float(m.group(0))
        return max(0.0, min(1.0, score)) >= threshold
    _v.__name__ = "llm_judge"
    return _v


def resolve_validator(v: Union[Validator, str, None],
                      test_case: TestCase) -> Validator:
    """Resolve a validator spec (callable, name, or None) to a callable.

    String names map to built-ins with reasonable defaults derived from the
    TestCase's expected_output.
    """
    if v is None:
        # Default: if expected_output provided, equals; else always-true
        if test_case.expected_output is None:
            return lambda r: True
        return equals(test_case.expected_output)
    if callable(v):
        return v
    if isinstance(v, str):
        name = v.lower()
        if name == "equals":
            return equals(test_case.expected_output)
        if name == "contains":
            return contains(test_case.expected_output)
        if name == "regex_match":
            if not isinstance(test_case.expected_output, str):
                raise ValidatorError("regex_match needs str expected_output")
            return regex_match(test_case.expected_output)
        if name == "json_schema_valid":
            schema = test_case.metadata.get("schema")
            return json_schema_valid(schema)
        if name == "llm_judge":
            judge = test_case.metadata.get("judge_llm")
            if not callable(judge):
                raise ValidatorError("llm_judge needs metadata['judge_llm'] callable")
            return llm_judge(
                judge,
                rubric=test_case.metadata.get("rubric",
                    "Rate the response on a 0–1 scale. Reply with only a number."),
                threshold=float(test_case.metadata.get("judge_threshold", 0.5)),
            )
        raise ValidatorError(f"Unknown validator name: {v!r}")
    raise ValidatorError(f"Invalid validator: {v!r}")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs a single TestCase against an agent/LLM and returns a TestResult.

    The `agent` callable is invoked with the test prompt and must return the
    model's output (string). It may raise; the error is captured.
    """

    def __init__(self,
                 agent: Callable[[str], Any],
                 default_latency_timeout_s: float = 60.0) -> None:
        if not callable(agent):
            raise EvaluatorError("agent must be callable")
        self.agent = agent
        self.default_latency_timeout_s = default_latency_timeout_s

    def run(self, test_case: TestCase) -> TestResult:
        validator = resolve_validator(test_case.validator, test_case)
        t0 = time.perf_counter()
        error: Optional[str] = None
        output: Any = None
        try:
            output = self.agent(test_case.prompt)
        except Exception as e:  # noqa: BLE001 — capture anything
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        latency = time.perf_counter() - t0

        passed = False
        score = 0.0
        if error is None:
            tmp = TestResult(
                test_id=test_case.id,
                passed=False,
                score=0.0,
                latency_s=latency,
                output=output,
                expected=test_case.expected_output,
                metadata=test_case.metadata,
            )
            try:
                passed = bool(validator(tmp))
            except Exception as e:  # noqa: BLE001
                error = f"validator raised: {type(e).__name__}: {e}"
            # If the validator returns a float between 0–1, treat as score
            if not passed:
                # Re-run to inspect return value if callable returned a number
                try:
                    ret = validator(tmp)
                    if isinstance(ret, (int, float)) and 0.0 <= float(ret) <= 1.0:
                        score = float(ret)
                        passed = score >= 0.5
                except Exception:
                    pass
            else:
                score = 1.0

        return TestResult(
            test_id=test_case.id,
            passed=passed and error is None,
            score=score if error is None else 0.0,
            latency_s=latency,
            output=output,
            expected=test_case.expected_output,
            error=error,
            metadata=test_case.metadata,
        )


# ---------------------------------------------------------------------------
# StreamingEvaluator
# ---------------------------------------------------------------------------

class StreamingEvaluator:
    """Wraps a streaming agent so latency can be measured per-token.

    Usage:
        se = StreamingEvaluator(streaming_agent)
        result = se.run(test_case)

    `streaming_agent` is expected to be a generator that yields tokens and
    returns the full string on StopIteration.
    """

    def __init__(self,
                 streaming_agent: Callable[[str], Any],
                 on_token: Optional[Callable[[str, float], None]] = None) -> None:
        self.streaming_agent = streaming_agent
        self.on_token = on_token
        self.token_latencies: List[float] = []

    def run(self, test_case: TestCase) -> TestResult:
        validator = resolve_validator(test_case.validator, test_case)
        t0 = time.perf_counter()
        first_token_at: Optional[float] = None
        last_token_at: Optional[float] = None
        chunks: List[str] = []
        error: Optional[str] = None
        try:
            gen = self.streaming_agent(test_case.prompt)
            for token in gen:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                chunks.append(str(token))
                self.token_latencies.append(now)
                if self.on_token:
                    try:
                        self.on_token(str(token), now - t0)
                    except Exception:
                        pass
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        output = "".join(chunks)
        time_to_first_token = (
            (first_token_at - t0) if first_token_at is not None else latency
        )
        tokens_per_second = (
            len(self.token_latencies) / latency
            if latency > 0 and self.token_latencies else 0.0
        )
        meta = dict(test_case.metadata)
        meta["time_to_first_token_s"] = time_to_first_token
        meta["tokens_per_second"] = tokens_per_second
        meta["num_tokens"] = len(self.token_latencies)
        tmp = TestResult(
            test_id=test_case.id,
            passed=False,
            score=0.0,
            latency_s=latency,
            output=output,
            expected=test_case.expected_output,
            metadata=meta,
        )
        passed = False
        score = 0.0
        if error is None:
            try:
                passed = bool(validator(tmp))
                if not passed:
                    ret = validator(tmp)
                    if isinstance(ret, (int, float)) and 0.0 <= float(ret) <= 1.0:
                        score = float(ret)
                        passed = score >= 0.5
                else:
                    score = 1.0
            except Exception as e:  # noqa: BLE001
                error = f"validator raised: {type(e).__name__}: {e}"
        return TestResult(
            test_id=test_case.id,
            passed=passed and error is None,
            score=score if error is None else 0.0,
            latency_s=latency,
            output=output,
            expected=test_case.expected_output,
            error=error,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# CoverageTracker
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Tracks which capabilities are tested across a suite.

    Capabilities are inferred from TestCase.metadata["capability"] (string or
    list). Use canonical labels like: "reasoning", "tool_use", "code_generation",
    "summarization", "translation", "math", "factual_qa", "json_formatting",
    "streaming", "rag", "planning".
    """

    DEFAULT_CAPABILITIES = [
        "reasoning",
        "tool_use",
        "code_generation",
        "summarization",
        "translation",
        "math",
        "factual_qa",
        "json_formatting",
        "rag",
        "planning",
    ]

    def __init__(self, capabilities: Optional[Sequence[str]] = None) -> None:
        self.capabilities: List[str] = list(capabilities or self.DEFAULT_CAPABILITIES)
        self._covered: Counter = Counter()

    def record(self, test_case: TestCase) -> None:
        cap = test_case.metadata.get("capability")
        if cap is None:
            return
        if isinstance(cap, str):
            self._covered[cap] += 1
        else:
            for c in cap:
                self._covered[c] += 1

    def record_suite(self, suite: Iterable[TestCase]) -> None:
        for tc in suite:
            self.record(tc)

    @property
    def covered(self) -> Dict[str, int]:
        return dict(self._covered)

    def coverage_ratio(self) -> float:
        if not self.capabilities:
            return 1.0
        return sum(1 for c in self.capabilities if self._covered.get(c, 0) > 0) / len(self.capabilities)

    def heatmap(self) -> Dict[str, Tuple[int, str]]:
        """Return {capability: (count, status)} where status ∈ {'covered','missing'}."""
        result: Dict[str, Tuple[int, str]] = {}
        for c in self.capabilities:
            count = self._covered.get(c, 0)
            result[c] = (count, "covered" if count > 0 else "missing")
        return result


# ---------------------------------------------------------------------------
# EvaluationSuite
# ---------------------------------------------------------------------------

@dataclass
class SuiteReport:
    """Aggregate metrics for an EvaluationSuite run."""
    name: str
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    avg_score: float
    results: List[TestResult] = field(default_factory=list)
    coverage: Dict[str, Tuple[int, str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


class EvaluationSuite:
    """A named collection of TestCases that can be run against an agent."""

    def __init__(self,
                 name: str,
                 test_cases: Optional[Sequence[TestCase]] = None,
                 capabilities: Optional[Sequence[str]] = None) -> None:
        self.name = name
        self.test_cases: List[TestCase] = list(test_cases or [])
        self.coverage_tracker = CoverageTracker(capabilities)
        if self.test_cases:
            self.coverage_tracker.record_suite(self.test_cases)

    def add(self, test_case: TestCase) -> None:
        self.test_cases.append(test_case)
        self.coverage_tracker.record(test_case)

    def extend(self, cases: Iterable[TestCase]) -> None:
        for c in cases:
            self.add(c)

    def run(self,
            agent: Callable[[str], Any],
            evaluator_cls: type = Evaluator) -> SuiteReport:
        if not self.test_cases:
            return SuiteReport(
                name=self.name, total=0, passed=0, failed=0, errors=0,
                pass_rate=0.0, avg_latency_s=0.0, p50_latency_s=0.0,
                p95_latency_s=0.0, avg_score=0.0, results=[],
                coverage=self.coverage_tracker.heatmap(),
            )
        ev = evaluator_cls(agent)
        results = [ev.run(tc) for tc in self.test_cases]
        return self._build_report(results)

    def run_streaming(self,
                      streaming_agent: Callable[[str], Any]) -> SuiteReport:
        ev = StreamingEvaluator(streaming_agent)
        results = [ev.run(tc) for tc in self.test_cases]
        return self._build_report(results)

    def _build_report(self, results: List[TestResult]) -> SuiteReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed and r.error is None)
        errors = sum(1 for r in results if r.error is not None)
        failed = total - passed - errors
        latencies = [r.latency_s for r in results]
        scores = [r.score for r in results]
        return SuiteReport(
            name=self.name,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=(passed / total) if total else 0.0,
            avg_latency_s=statistics.fmean(latencies) if latencies else 0.0,
            p50_latency_s=_percentile(latencies, 50),
            p95_latency_s=_percentile(latencies, 95),
            avg_score=statistics.fmean(scores) if scores else 0.0,
            results=results,
            coverage=self.coverage_tracker.heatmap(),
        )


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Benchmark (multi-suite comparison)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    """One row in a benchmark comparison: a single (suite, agent) pair."""
    label: str
    suite_name: str
    report: SuiteReport


@dataclass
class BenchmarkReport:
    """Side-by-side comparison of multiple BenchmarkEntries on the same suites."""
    name: str
    entries: List[BenchmarkEntry]
    suite_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entries": [
                {
                    "label": e.label,
                    "suite_name": e.suite_name,
                    "pass_rate": e.report.pass_rate,
                    "avg_score": e.report.avg_score,
                    "avg_latency_s": e.report.avg_latency_s,
                    "p50_latency_s": e.report.p50_latency_s,
                    "p95_latency_s": e.report.p95_latency_s,
                    "passed": e.report.passed,
                    "failed": e.report.failed,
                    "errors": e.report.errors,
                    "total": e.report.total,
                }
                for e in self.entries
            ],
            "suite_names": self.suite_names,
        }


class Benchmark:
    """Run the same suites against multiple agents and produce a comparison.

    Example:
        bench = Benchmark("gpt-4-vs-claude", suites=[s1, s2])
        bench.add_run("gpt-4", lambda p: openai_call(p))
        bench.add_run("claude", lambda p: anthropic_call(p))
        report = bench.run()
    """

    def __init__(self,
                 name: str,
                 suites: Sequence[EvaluationSuite]) -> None:
        self.name = name
        self.suites: List[EvaluationSuite] = list(suites)
        self._runs: List[Tuple[str, Callable[[str], Any]]] = []
        self._baseline: Optional[str] = None

    def add_run(self, label: str, agent: Callable[[str], Any]) -> None:
        self._runs.append((label, agent))

    def set_baseline(self, label: str) -> None:
        if label not in [l for l, _ in self._runs]:
            raise EvaluatorError(f"Unknown baseline label: {label!r}")
        self._baseline = label

    def run(self) -> BenchmarkReport:
        entries: List[BenchmarkEntry] = []
        for label, agent in self._runs:
            for suite in self.suites:
                report = suite.run(agent)
                entries.append(BenchmarkEntry(label=label,
                                              suite_name=suite.name,
                                              report=report))
        return BenchmarkReport(
            name=self.name,
            entries=entries,
            suite_names=[s.name for s in self.suites],
        )


def diff_benchmark(report: BenchmarkReport,
                    baseline_label: Optional[str] = None) -> Dict[str, Any]:
    """Return per-suite deltas vs the baseline (defaults to first run).

    Returns:
        {
          suite_name: {
            label: {"pass_rate": delta, "avg_score": delta, "avg_latency_s": delta},
            ...
          },
          ...
        }
    """
    if not report.entries:
        return {}
    base_label = baseline_label or report.entries[0].label
    base_by_suite = {
        e.suite_name: e.report for e in report.entries if e.label == base_label
    }
    diff: Dict[str, Any] = {}
    for e in report.entries:
        diff.setdefault(e.suite_name, {})[e.label] = {
            "pass_rate": e.report.pass_rate,
            "avg_score": e.report.avg_score,
            "avg_latency_s": e.report.avg_latency_s,
        }
    for suite_name, by_label in diff.items():
        base = base_by_suite.get(suite_name)
        if not base:
            continue
        for label, vals in by_label.items():
            if label == base_label:
                vals["delta_vs_baseline"] = {"pass_rate": 0.0, "avg_score": 0.0, "avg_latency_s": 0.0}
            else:
                vals["delta_vs_baseline"] = {
                    "pass_rate": vals["pass_rate"] - base.pass_rate,
                    "avg_score": vals["avg_score"] - base.avg_score,
                    "avg_latency_s": vals["avg_latency_s"] - base.avg_latency_s,
                }
    return diff


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>tiny-evaluator report — {suite_name}</title>
<style>
  :root {{
    --bg: #0f1117; --fg: #e6e6e6; --muted: #9aa3b2; --line: #222732;
    --ok: #21c97a; --fail: #ff6b6b; --err: #f5a524; --chip: #1c2030;
    --accent: #6ea8ff; --accent2: #b78bff;
  }}
  body {{ background: var(--bg); color: var(--fg); font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; }}
  h1, h2, h3 {{ margin: 0 0 12px; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 16px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: 28px; }}
  .meta {{ color: var(--muted); margin-bottom: 16px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
  .stat {{ background: var(--chip); padding: 12px; border-radius: 8px; }}
  .stat .v {{ font-size: 22px; font-weight: 600; }}
  .stat .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
  table {{ border-collapse: collapse; width: 100%; background: var(--chip); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
  th {{ background: #161a26; color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
  tr:last-child td {{ border-bottom: none; }}
  .pass {{ color: var(--ok); }}
  .fail {{ color: var(--fail); }}
  .err  {{ color: var(--err); }}
  details {{ background: var(--chip); border-radius: 6px; padding: 8px 12px; margin-top: 8px; }}
  details summary {{ cursor: pointer; color: var(--muted); }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #0a0d14; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  .bar {{ background: #0a0d14; border-radius: 4px; height: 18px; overflow: hidden; }}
  .bar > span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}
  .heat {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }}
  .heat > div {{ background: var(--chip); padding: 10px; border-radius: 6px; }}
  .heat .on  {{ color: var(--ok); }}
  .heat .off {{ color: var(--fail); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; background: var(--chip); color: var(--muted); margin-right: 4px; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
  <h1>📊 tiny-evaluator report</h1>
  <div class="meta">Suite <strong>{suite_name}</strong> · {total} test cases · {pass_rate:.1%} pass rate · generated {ts}</div>

  <div class="summary">
    <div class="stat"><div class="v pass">{passed}</div><div class="k">passed</div></div>
    <div class="stat"><div class="v fail">{failed}</div><div class="k">failed</div></div>
    <div class="stat"><div class="v err">{errors}</div><div class="k">errors</div></div>
    <div class="stat"><div class="v">{avg_score:.2f}</div><div class="k">avg score</div></div>
    <div class="stat"><div class="v">{avg_latency_s:.2f}s</div><div class="k">avg latency</div></div>
    <div class="stat"><div class="v">{p95_latency_s:.2f}s</div><div class="k">p95 latency</div></div>
  </div>

  <h2>Pass rate</h2>
  <div class="bar" title="{pass_rate:.1%}"><span style="width:{pass_rate:.1%}"></span></div>

  {benchmark_section}

  <h2>Capability coverage</h2>
  <div class="heat">{coverage_html}</div>

  <h2>Results</h2>
  <table>
    <thead><tr><th>#</th><th>ID</th><th>Status</th><th>Score</th><th>Latency</th><th>Capabilities</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <h2>Failure details</h2>
  {details_html}

  <p style="color:var(--muted);margin-top:32px;font-size:12px;">
    Generated by <a href="https://github.com/hussain-alsaibai/tiny-evaluator">tiny-evaluator</a>.
  </p>
</body>
</html>
"""


def _benchmark_table_html(report: "BenchmarkReport") -> str:
    """Render a comparison table for a BenchmarkReport."""
    if not report.entries:
        return ""
    labels = sorted({e.label for e in report.entries})
    suites = report.suite_names
    rows: List[str] = []
    rows.append("<table><thead><tr><th>Suite</th>")
    for lbl in labels:
        rows.append(f"<th>{_html.escape(lbl)}<br><span class='meta'>pass · score · p95</span></th>")
    rows.append("</tr></thead><tbody>")
    # Build a lookup
    lookup: Dict[Tuple[str, str], SuiteReport] = {
        (e.suite_name, e.label): e.report for e in report.entries
    }
    for s in suites:
        rows.append(f"<tr><td><strong>{_html.escape(s)}</strong></td>")
        for lbl in labels:
            r = lookup.get((s, lbl))
            if r is None:
                rows.append("<td>—</td>")
            else:
                rows.append(
                    f"<td>{r.passed}/{r.total} · {r.avg_score:.2f} · "
                    f"{r.p95_latency_s:.2f}s</td>"
                )
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "<h2>Benchmark comparison</h2>" + "".join(rows)


def _coverage_html(heatmap: Dict[str, Tuple[int, str]]) -> str:
    if not heatmap:
        return "<div class='meta'>No capabilities tracked.</div>"
    out = []
    for cap, (count, status) in heatmap.items():
        cls = "on" if status == "covered" else "off"
        out.append(
            f"<div><div class='{cls}'><strong>{_html.escape(cap)}</strong></div>"
            f"<div class='meta'>{count} test{'s' if count != 1 else ''}</div></div>"
        )
    return "".join(out)


def _row_html(i: int, r: TestResult) -> str:
    if r.error:
        status = f"<span class='err'>ERROR</span>"
    elif r.passed:
        status = f"<span class='pass'>PASS</span>"
    else:
        status = f"<span class='fail'>FAIL</span>"
    caps = r.metadata.get("capability", "")
    if isinstance(caps, list):
        caps = ", ".join(str(c) for c in caps)
    return (
        f"<tr><td>{i}</td><td>{_html.escape(r.test_id)}</td>"
        f"<td>{status}</td><td>{r.score:.2f}</td>"
        f"<td>{r.latency_s:.2f}s</td>"
        f"<td>{_html.escape(str(caps))}</td></tr>"
    )


def _details_html(results: Sequence[TestResult]) -> str:
    failures = [r for r in results if not r.passed or r.error]
    if not failures:
        return "<div class='meta'>No failures 🎉</div>"
    parts = []
    for r in failures:
        meta = r.metadata or {}
        cap = meta.get("capability", "")
        if isinstance(cap, list):
            cap = ", ".join(str(c) for c in cap)
        badges = "".join(
            f"<span class='badge'>{_html.escape(str(k))}={_html.escape(str(v))}</span>"
            for k, v in meta.items() if k != "capability"
        )
        body = (
            f"<p><strong>Prompt:</strong></p><pre>{_html.escape(str(r.output)[:2000] if not r.error else '(agent raised)')}</pre>"
            f"<p><strong>Expected:</strong></p><pre>{_html.escape(repr(r.expected))}</pre>"
            f"<p><strong>Error:</strong></p><pre>{_html.escape((r.error or '')[:2000])}</pre>"
            f"<p>{badges}</p>"
        )
        parts.append(
            f"<details><summary>{_html.escape(r.test_id)} "
            f"<span class='{'err' if r.error else 'fail'}'>"
            f"{'ERROR' if r.error else 'FAIL'}</span></summary>{body}</details>"
        )
    return "".join(parts)


def generate_html_report(report_or_results: Union["SuiteReport", "BenchmarkReport", Sequence[TestResult]],
                         output_path: Optional[str] = None) -> str:
    """Render an HTML report. Returns the HTML string.

    Args:
        report_or_results: a SuiteReport, BenchmarkReport, or list of TestResults
        output_path:      if given, also writes the HTML to this file

    For a BenchmarkReport, the report includes a comparison table across all
    entries in addition to the per-test results.
    """
    if isinstance(report_or_results, SuiteReport):
        rep = report_or_results
        benchmark_section = ""
    elif isinstance(report_or_results, BenchmarkReport):
        # Flatten all entries' results into one suite report named after the
        # benchmark, plus emit a comparison table at the top.
        all_results: List[TestResult] = []
        for e in report_or_results.entries:
            # tag each result with the run label so it shows up in the table
            for r in e.report.results:
                tagged = TestResult(
                    test_id=f"[{e.label}] {r.test_id}",
                    passed=r.passed,
                    score=r.score,
                    latency_s=r.latency_s,
                    output=r.output,
                    expected=r.expected,
                    error=r.error,
                    metadata={**(r.metadata or {}), "benchmark_label": e.label},
                )
                all_results.append(tagged)
        total = len(all_results)
        passed = sum(1 for r in all_results if r.passed and r.error is None)
        errors = sum(1 for r in all_results if r.error is not None)
        failed = total - passed - errors
        latencies = [r.latency_s for r in all_results]
        scores = [r.score for r in all_results]
        rep = SuiteReport(
            name=report_or_results.name,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=(passed / total) if total else 0.0,
            avg_latency_s=statistics.fmean(latencies) if latencies else 0.0,
            p50_latency_s=_percentile(latencies, 50),
            p95_latency_s=_percentile(latencies, 95),
            avg_score=statistics.fmean(scores) if scores else 0.0,
            results=all_results,
            coverage={},
        )
        benchmark_section = _benchmark_table_html(report_or_results)
    else:
        results = list(report_or_results)
        total = len(results)
        passed = sum(1 for r in results if r.passed and r.error is None)
        errors = sum(1 for r in results if r.error is not None)
        failed = total - passed - errors
        latencies = [r.latency_s for r in results]
        scores = [r.score for r in results]
        cov: Counter = Counter()
        for r in results:
            c = (r.metadata or {}).get("capability")
            if isinstance(c, str):
                cov[c] += 1
            elif isinstance(c, list):
                for x in c:
                    cov[x] += 1
        rep = SuiteReport(
            name="ad-hoc",
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=(passed / total) if total else 0.0,
            avg_latency_s=statistics.fmean(latencies) if latencies else 0.0,
            p50_latency_s=_percentile(latencies, 50),
            p95_latency_s=_percentile(latencies, 95),
            avg_score=statistics.fmean(scores) if scores else 0.0,
            results=results,
            coverage={k: (v, "covered" if v > 0 else "missing") for k, v in cov.items()},
        )
        benchmark_section = ""

    rows_html = "".join(_row_html(i, r) for i, r in enumerate(rep.results, start=1))
    details_html = _details_html(rep.results)
    coverage_html = _coverage_html(rep.coverage)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    html = _HTML_TEMPLATE.format(
        suite_name=_html.escape(rep.name),
        total=rep.total,
        passed=rep.passed,
        failed=rep.failed,
        errors=rep.errors,
        pass_rate=rep.pass_rate,
        avg_score=rep.avg_score,
        avg_latency_s=rep.avg_latency_s,
        p50_latency_s=rep.p50_latency_s,
        p95_latency_s=rep.p95_latency_s,
        ts=ts,
        coverage_html=coverage_html,
        rows_html=rows_html,
        details_html=details_html,
        benchmark_section=benchmark_section,
    )
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    # dataclasses
    "TestCase",
    "TestResult",
    "SuiteReport",
    "BenchmarkEntry",
    "BenchmarkReport",
    # core classes
    "Evaluator",
    "EvaluationSuite",
    "StreamingEvaluator",
    "CoverageTracker",
    "Benchmark",
    # validators
    "equals",
    "contains",
    "regex_match",
    "json_schema_valid",
    "llm_judge",
    "resolve_validator",
    # utilities
    "diff_benchmark",
    "generate_html_report",
    # errors
    "EvaluatorError",
    "ValidatorError",
]
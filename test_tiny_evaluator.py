"""Tests for tiny_evaluator.

Run with:
    pytest test_tiny_evaluator.py
or:
    python -m unittest test_tiny_evaluator.py
"""

from __future__ import annotations

import json
import tempfile
import unittest

from tiny_evaluator import (
    Benchmark,
    BenchmarkReport,
    CoverageTracker,
    Evaluator,
    EvaluationSuite,
    StreamingEvaluator,
    TestCase,
    TestResult,
    contains,
    diff_benchmark,
    equals,
    generate_html_report,
    json_schema_valid,
    llm_judge,
    regex_match,
    resolve_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mock_agent(response_map=None, default="ok"):
    """Build a simple agent that responds based on a {prompt_substr: response} map."""
    response_map = response_map or {}

    def _agent(prompt: str) -> str:
        for key, val in response_map.items():
            if key in prompt:
                return val
        return default

    return _agent


def mock_streaming_agent(response_map=None, default="ok"):
    response_map = response_map or {}

    def _gen(prompt: str):
        text = default
        for key, val in response_map.items():
            if key in prompt:
                text = val
                break
        for tok in text.split(" "):
            yield tok + " "

    return _gen


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidators(unittest.TestCase):

    def test_equals(self):
        v = equals("hello")
        r_pass = TestResult(test_id="t", passed=False, score=0, latency_s=0, output="hello")
        r_fail = TestResult(test_id="t", passed=False, score=0, latency_s=0, output="bye")
        self.assertTrue(v(r_pass))
        self.assertFalse(v(r_fail))

    def test_contains_string(self):
        v = contains("foo")
        self.assertTrue(v(TestResult("t", False, 0, 0, output="foobar")))
        self.assertFalse(v(TestResult("t", False, 0, 0, output="baz")))

    def test_contains_sequence(self):
        v = contains(["a", "b"])
        self.assertTrue(v(TestResult("t", False, 0, 0, output="a, b, c")))
        self.assertFalse(v(TestResult("t", False, 0, 0, output="a, c")))

    def test_regex_match(self):
        v = regex_match(r"\d{3}-\d{4}")
        self.assertTrue(v(TestResult("t", False, 0, 0, output="Call 555-1234 now")))
        self.assertFalse(v(TestResult("t", False, 0, 0, output="no phone here")))

    def test_json_schema_valid_no_schema(self):
        v = json_schema_valid()
        self.assertTrue(v(TestResult("t", False, 0, 0, output='{"a":1}')))
        self.assertFalse(v(TestResult("t", False, 0, 0, output="not json")))

    def test_json_schema_valid_with_schema(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        v = json_schema_valid(schema)
        self.assertTrue(v(TestResult("t", False, 0, 0,
                                     output=json.dumps({"name": "Alice", "age": 30}))))
        self.assertFalse(v(TestResult("t", False, 0, 0,
                                      output=json.dumps({"name": "Alice"}))))
        self.assertFalse(v(TestResult("t", False, 0, 0,
                                      output=json.dumps({"name": "Alice", "age": "thirty"}))))

    def test_resolve_validator_defaults(self):
        tc = TestCase(id="t", prompt="p", expected_output="ok")
        v = resolve_validator(None, tc)
        self.assertTrue(v(TestResult("t", False, 0, 0, output="ok")))
        self.assertFalse(v(TestResult("t", False, 0, 0, output="nope")))

    def test_resolve_validator_unknown_string(self):
        tc = TestCase(id="t", prompt="p")
        with self.assertRaises(Exception):
            resolve_validator("not_a_real_validator", tc)


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator(unittest.TestCase):

    def test_run_passing(self):
        agent = mock_agent({"1+1": "2"})
        ev = Evaluator(agent)
        tc = TestCase(id="math", prompt="1+1?", expected_output="2")
        r = ev.run(tc)
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.output, "2")
        self.assertIsNone(r.error)
        self.assertGreaterEqual(r.latency_s, 0)

    def test_run_failing(self):
        agent = mock_agent({"capital": "Paris"})
        ev = Evaluator(agent)
        tc = TestCase(id="geo", prompt="capital of Japan?", expected_output="Tokyo")
        r = ev.run(tc)
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.0)

    def test_run_captures_exception(self):
        def bad_agent(p):
            raise RuntimeError("model down")

        ev = Evaluator(bad_agent)
        tc = TestCase(id="x", prompt="hi")
        r = ev.run(tc)
        self.assertFalse(r.passed)
        self.assertIn("RuntimeError", r.error or "")
        self.assertIsNone(r.output)

    def test_non_callable_agent_raises(self):
        with self.assertRaises(Exception):
            Evaluator("not callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# StreamingEvaluator tests
# ---------------------------------------------------------------------------

class TestStreamingEvaluator(unittest.TestCase):

    def test_streaming_basic(self):
        agent = mock_streaming_agent({"hi": "hello world"})
        se = StreamingEvaluator(agent)
        tc = TestCase(id="s", prompt="hi", expected_output="hello world ")
        r = se.run(tc)
        self.assertTrue(r.passed)
        # "hello world" split on " " yields ["hello", "world"] → 2 tokens
        self.assertEqual(r.metadata["num_tokens"], 2)
        self.assertIn("time_to_first_token_s", r.metadata)

    def test_streaming_captures_error(self):
        def bad_gen(p):
            yield "ok"
            raise ValueError("oops")

        se = StreamingEvaluator(bad_gen)
        tc = TestCase(id="x", prompt="hi")
        r = se.run(tc)
        self.assertFalse(r.passed)
        self.assertIn("ValueError", r.error or "")


# ---------------------------------------------------------------------------
# Suite / Coverage tests
# ---------------------------------------------------------------------------

class TestEvaluationSuite(unittest.TestCase):

    def test_suite_run_collects_results(self):
        suite = EvaluationSuite(
            name="math",
            test_cases=[
                TestCase(id="q1", prompt="2+2", expected_output="4"),
                TestCase(id="q2", prompt="3+3", expected_output="7"),  # wrong
            ],
        )
        agent = mock_agent({"2+2": "4", "3+3": "6"})
        rep = suite.run(agent)
        self.assertEqual(rep.total, 2)
        self.assertEqual(rep.passed, 1)
        self.assertEqual(rep.failed, 1)
        self.assertEqual(rep.errors, 0)
        self.assertAlmostEqual(rep.pass_rate, 0.5)
        self.assertGreaterEqual(rep.avg_latency_s, 0)

    def test_suite_records_coverage(self):
        suite = EvaluationSuite(
            name="mixed",
            test_cases=[
                TestCase(id="a", prompt="p1", expected_output="ok",
                         metadata={"capability": "math"}),
                TestCase(id="b", prompt="p2", expected_output="ok",
                         metadata={"capability": ["math", "reasoning"]}),
            ],
        )
        self.assertEqual(suite.coverage_tracker.coverage_ratio(), 2 / len(CoverageTracker.DEFAULT_CAPABILITIES) + 0)  # 2 of 10

    def test_empty_suite(self):
        suite = EvaluationSuite(name="empty")
        rep = suite.run(mock_agent({}))
        self.assertEqual(rep.total, 0)
        self.assertEqual(rep.pass_rate, 0.0)

    def test_add_extend(self):
        suite = EvaluationSuite(name="x")
        suite.add(TestCase(id="1", prompt="p"))
        suite.extend([TestCase(id="2", prompt="p")])
        self.assertEqual(len(suite.test_cases), 2)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmark(unittest.TestCase):

    def test_benchmark_compare_two_agents(self):
        suite = EvaluationSuite(name="truth", test_cases=[
            TestCase(id="t1", prompt="hi", expected_output="hello"),
            TestCase(id="t2", prompt="bye", expected_output="goodbye"),
        ])
        bench = Benchmark("compare", suites=[suite])
        bench.add_run("good", mock_agent({"hi": "hello", "bye": "goodbye"}))
        bench.add_run("bad", mock_agent({"hi": "nope", "bye": "wrong"}))
        report = bench.run()
        self.assertEqual(len(report.entries), 2)
        labels = {e.label for e in report.entries}
        self.assertEqual(labels, {"good", "bad"})
        diff = diff_benchmark(report, baseline_label="good")
        self.assertIn("truth", diff)
        self.assertAlmostEqual(diff["truth"]["good"]["pass_rate"], 1.0)
        self.assertAlmostEqual(diff["truth"]["bad"]["pass_rate"], 0.0)

    def test_benchmark_no_runs(self):
        bench = Benchmark("empty", suites=[EvaluationSuite(name="x")])
        report = bench.run()
        self.assertEqual(len(report.entries), 0)


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------

class TestHtmlReport(unittest.TestCase):

    def test_generate_html_report_from_results(self):
        results = [
            TestResult("a", True, 1.0, 0.1, output="yes", expected="yes"),
            TestResult("b", False, 0.0, 0.2, output="no", expected="yes",
                       error="boom"),
        ]
        html = generate_html_report(results)
        self.assertIn("<!doctype html>", html.lower())
        self.assertIn("tiny-evaluator", html)
        self.assertIn("a", html)
        self.assertIn("b", html)
        self.assertIn("PASS", html)
        self.assertIn("ERROR", html)

    def test_generate_html_report_to_file(self):
        rep = EvaluationSuite(name="demo", test_cases=[
            TestCase(id="x", prompt="p", expected_output="ok"),
        ]).run(mock_agent({"p": "ok"}))
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        generate_html_report(rep, output_path=path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("demo", content)
        self.assertIn("PASS", content)

    def test_generate_html_report_from_benchmark(self):
        suite = EvaluationSuite(name="s1", test_cases=[
            TestCase(id="t1", prompt="hi", expected_output="hello"),
        ])
        bench = Benchmark("cmp", suites=[suite])
        bench.add_run("good", mock_agent({"hi": "hello"}))
        bench.add_run("bad",  mock_agent({"hi": "nope"}))
        report = bench.run()
        html = generate_html_report(report)
        self.assertIn("Benchmark comparison", html)
        self.assertIn("good", html)
        self.assertIn("bad", html)


# ---------------------------------------------------------------------------
# LLM-judge tests
# ---------------------------------------------------------------------------

class TestLLMJudge(unittest.TestCase):

    def test_llm_judge_passing(self):
        def judge(prompt):
            return "0.9"

        v = llm_judge(judge, threshold=0.5)
        r = TestResult("t", False, 0, 0, output="great answer", expected="ok")
        self.assertTrue(v(r))

    def test_llm_judge_failing(self):
        def judge(prompt):
            return "0.2"

        v = llm_judge(judge, threshold=0.5)
        r = TestResult("t", False, 0, 0, output="bad answer", expected="ok")
        self.assertFalse(v(r))

    def test_llm_judge_unparseable(self):
        def judge(prompt):
            return "I don't know"

        v = llm_judge(judge)
        r = TestResult("t", False, 0, 0, output="x", expected="y")
        self.assertFalse(v(r))


if __name__ == "__main__":
    unittest.main()
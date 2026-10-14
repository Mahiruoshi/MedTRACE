from __future__ import annotations

import importlib.util
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from medtrace.data.tasks import SurvivalTarget, Task, survival_bin, survival_edges


def _have(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def brute_c_index(times, events, risks):
    num = den = 0.0
    n = len(times)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[i] < times[j] or (times[i] == times[j] and not events[j]):
                den += 1
                num += 1.0 if risks[i] > risks[j] else 0.5 if risks[i] == risks[j] else 0.0
    return None if den == 0 else num / den


@unittest.skipUnless(_have("numpy", "scipy"), "numpy and scipy are required")
class TestCIndex(unittest.TestCase):
    def setUp(self):
        from medtrace.eval.metrics import c_index

        self.c_index = c_index

    def test_fixed_data_with_ties_and_censoring(self):
        times = [5, 5, 3, 8, 8, 10, 1, 3]
        events = [1, 0, 1, 1, 0, 0, 1, 1]
        risks = [2, 1, 2, 0, 1, 0, 2, 3]
        self.assertAlmostEqual(self.c_index(times, events, risks), brute_c_index(times, events, risks), places=12)

    def test_tie_conventions(self):
        self.assertEqual(self.c_index([2, 2], [1, 0], [1, 0]), 1.0)
        self.assertEqual(self.c_index([2, 2], [1, 0], [0, 1]), 0.0)
        self.assertEqual(self.c_index([2, 2], [1, 0], [1, 1]), 0.5)
        self.assertIsNone(self.c_index([2, 2], [1, 1], [1, 0]))

    def test_random_against_brute_force(self):
        rng = random.Random(1234)
        for _ in range(300):
            n = rng.randint(2, 12)
            times = [rng.randint(1, 6) for _ in range(n)]
            events = [int(rng.random() < 0.6) for _ in range(n)]
            risks = [rng.randint(0, 3) for _ in range(n)]
            want = brute_c_index(times, events, risks)
            got = self.c_index(times, events, risks)
            if want is None:
                self.assertIsNone(got)
            else:
                self.assertAlmostEqual(got, want, places=12)

    def test_perfect_order_and_reversed(self):
        times, events = [1, 2, 3, 4], [1, 1, 1, 1]
        self.assertEqual(self.c_index(times, events, [3, 2, 1, 0]), 1.0)
        self.assertEqual(self.c_index(times, events, [0, 1, 2, 3]), 0.0)

    def test_all_censored(self):
        self.assertIsNone(self.c_index([1, 2, 3], [0, 0, 0], [2, 1, 0]))


class TestSurvivalBins(unittest.TestCase):
    def test_edges_and_bins_one_to_nine(self):
        times = list(range(1, 10))
        edges = survival_edges(times, [1] * 9, 3)
        self.assertEqual(len(edges), 2)
        self.assertAlmostEqual(edges[0], 11 / 3)
        self.assertAlmostEqual(edges[1], 19 / 3)
        self.assertEqual([survival_bin(t, edges) for t in times], [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_censored_times_do_not_move_the_edges(self):
        a = survival_edges(list(range(1, 10)), [1] * 9, 3)
        b = survival_edges(list(range(1, 10)) + [0.5, 100.0], [1] * 9 + [0, 0], 3)
        self.assertEqual(a, b)

    def test_an_edge_at_a_time_goes_up(self):
        self.assertEqual(survival_bin(5.0, (5.0,)), 1)
        self.assertEqual(survival_bin(4.999, (5.0,)), 0)

    def test_too_few_events_or_an_empty_bin_raise(self):
        with self.assertRaises(ValueError):
            survival_edges([1, 2, 3], [1, 1, 0], 3)
        with self.assertRaises(ValueError):
            survival_edges([1, 1, 1, 2, 3, 4], [1] * 6, 3)
        with self.assertRaises(ValueError):
            survival_edges([1, 2, float("nan"), 4], [1] * 4, 2)


class TestTask(unittest.TestCase):
    STAGES = ("Short", "Medium", "Long")

    def survival(self):
        return Task(kind="survival", labels=self.STAGES)

    def test_survival_correctness_is_three_valued(self):
        t = self.survival()
        ev = SurvivalTarget(time=10.0, event=1, bin=1)
        self.assertEqual([t.is_correct(p, ev) for p in (0, 1, 2, None)], [0, 1, 0, 0])
        cens_mid = SurvivalTarget(time=10.0, event=0, bin=1)
        self.assertEqual([t.is_correct(p, cens_mid) for p in (0, 1, 2, None)], [0, None, None, 0])
        cens_last = SurvivalTarget(time=99.0, event=0, bin=2)
        self.assertEqual([t.is_correct(p, cens_last) for p in (0, 1, 2, None)], [0, 0, 1, 0])
        cens_first = SurvivalTarget(time=1.0, event=0, bin=0)
        self.assertEqual([t.is_correct(p, cens_first) for p in (0, 1, 2)], [None, None, None])

    def test_determined(self):
        t = self.survival()
        self.assertTrue(t.determined(SurvivalTarget(10.0, 1, 0)))
        self.assertTrue(t.determined(SurvivalTarget(10.0, 0, 2)))
        self.assertFalse(t.determined(SurvivalTarget(10.0, 0, 1)))
        self.assertFalse(t.determined(SurvivalTarget(10.0, 0, 0)))
        b = Task(kind="binary", labels=("No", "Yes"))
        self.assertTrue(b.determined(0) and b.determined(1))
        self.assertEqual([b.is_correct(p, 1) for p in (0, 1, None)], [0, 1, 0])
        self.assertEqual([b.is_correct(p, 0) for p in (0, 1, None)], [1, 0, 0])

    def test_risk_order(self):
        t = self.survival()
        self.assertEqual([t.risk(i) for i in range(3)], [2.0, 1.0, 0.0])

    def test_alias_collisions_raise(self):
        with self.assertRaises(ValueError):
            Task(kind="binary", labels=("No", "Yes"), aliases={"Yes": ("no",)})
        with self.assertRaises(ValueError):
            Task(kind="binary", labels=("No", "Yes"), aliases={"No": ("negative",), "Yes": ("Negative",)})
        with self.assertRaises(ValueError):
            Task(kind="binary", labels=("No", "Yes"), aliases={"Yes": ("-",)})
        with self.assertRaises(ValueError):
            Task(kind="binary", labels=("No", "Yes"), aliases={"Maybe": ("perhaps",)})
        ok = Task(kind="binary", labels=("No", "Yes"), aliases={"Yes": ("positive",), "No": ("negative",)})
        self.assertEqual((ok.parse_label("POSITIVE"), ok.parse_label("negative"), ok.parse_label("x")), (1, 0, None))

    def test_answer_prefix_needs_a_separator(self):
        with self.assertRaises(ValueError):
            Task(kind="binary", labels=("No", "Yes"), answer_prefix="Answer")
        Task(kind="binary", labels=("No", "Yes"), answer_prefix="Answer:")


@unittest.skipUnless(_have("yaml"), "pyyaml is required")
class TestParsing(unittest.TestCase):
    def setUp(self):
        from medtrace.generation.parsing import parse

        self.parse = parse
        self.task = Task(kind="survival", labels=("Stage I", "Stage II", "Stage III"),
                         step_names=("Findings",), answer_prefix="Answer:")

    def pred(self, answer_line: str):
        return self.parse(f"Step 1 - Findings: one finding is described.\n\n{answer_line}", self.task).pred

    def test_longer_label_first(self):
        self.assertEqual(self.pred("Answer: Stage II - moderate extent."), 1)
        self.assertEqual(self.pred("Answer: Stage I - localized."), 0)
        self.assertEqual(self.pred("Answer: Stage III"), 2)

    def test_echoed_placeholder_is_stripped(self):
        self.assertEqual(self.pred("Answer: [Stage I / Stage II / Stage III] Stage III - extensive."), 2)

    def test_two_stage_hedge_is_unparseable(self):
        self.assertIsNone(self.pred("Answer: Stage I or Stage II - unclear."))

    def test_steps_and_answer_line(self):
        p = self.parse("Step 1 - Findings: one finding.\n\nAnswer: Stage II - moderate extent.", self.task)
        self.assertEqual(len(p.steps), self.task.n_steps)
        self.assertEqual(p.answer, "Stage II")


@unittest.skipUnless(_have("yaml"), "pyyaml is required")
class TestFrameLine(unittest.TestCase):
    PUB = {"pid": "p01", "age": "", "T": 9}

    @staticmethod
    def task(frame_line: str, question: str = "Which class fits the series?") -> Task:
        return Task(kind="binary", labels=("Negative", "Positive"), question=question, system="Read the frames.",
                    user="{m} of {t} frames\n{slice_lines}\n{question}", frame_line=frame_line,
                    step_names=("Findings",), answer_prefix="Answer:")

    def test_frame_line_renders_one_line_per_shown_frame(self):
        from medtrace.generation.prompts import prompt_record

        rec = prompt_record(self.PUB, self.task("[Frame {j}] position {i}/{t}"), "CT", [4, 0])
        self.assertEqual(rec["question"], "2 of 9 frames\n[Frame 1] position 5/9\n[Frame 2] position 1/9\n"
                                          "Which class fits the series?")
        self.assertEqual(rec["frame_ids"], ["p01/s004", "p01/s000"])

    def test_unknown_placeholder_in_frame_line_raises(self):
        from medtrace.generation.prompts import PromptError, prompt_record, required_prompt_fields

        bad = self.task("[Frame {j}] of {pid}")
        with self.assertRaises(PromptError):
            required_prompt_fields(bad)
        with self.assertRaises(PromptError):
            prompt_record(self.PUB, bad, "CT", [4, 0])
        self.assertIn("prompts.frame_line", required_prompt_fields(self.task("[Frame {j}]")))

    def test_frame_line_text_outside_slice_lines_is_a_leak(self):
        from medtrace.generation.prompts import LeakError, prompt_record

        with self.assertRaises(LeakError):
            prompt_record(self.PUB, self.task("[Frame {j}] position {i}/{t}",
                                              question="Is [Frame 3] position 2/9 the key one?"), "CT", [4, 0])


@unittest.skipUnless(_have("yaml"), "pyyaml is required")
class TestRunDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="medtrace_units_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_output_is_not_done(self):
        from medtrace.pipeline.rundir import RunDir, write_json

        rd = RunDir(self.tmp / "runs", "exp", "g", "v", 1)
        cfg = {"a": 1}
        self.assertEqual(rd.begin("distill", cfg), "new")
        write_json(rd.signals, {"signals": []})
        rd.finish("distill", [rd.signals])
        self.assertEqual(rd.stale_reason("distill", cfg), "")
        self.assertEqual(rd.begin("distill", cfg), "done")
        rd.signals.unlink()
        self.assertIn("missing", rd.stale_reason("distill", cfg))
        self.assertEqual(rd.begin("distill", cfg), "new")
        self.assertTrue(any(p.name.startswith("distill.superseded-") for p in rd.root.iterdir()))

    def test_cache_inputs_are_keyed_by_the_cache_root(self):
        from medtrace.pipeline.rundir import RunDir

        keys = []
        for name in ("cache_a", "cache_b"):
            meta = self.tmp / name / "features" / "tag" / "meta.json"
            meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({"d_h": 4}), encoding="utf-8")
            rd = RunDir(self.tmp / "runs", "exp", "g", "v", 1, cache_root=self.tmp / name)
            keys.append(rd.hash_inputs([meta]))
            self.assertEqual(rd.resolve("cache:features/tag/meta.json"), self.tmp / name / "features/tag/meta.json")
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(list(keys[0]), ["cache:features/tag/meta.json"])


@unittest.skipUnless(_have("yaml"), "pyyaml is required")
class TestConfigPids(unittest.TestCase):
    def test_forms(self):
        from medtrace.pipeline.stages import config_pids

        self.assertIsNone(config_pids({"pids": None}))
        self.assertIsNone(config_pids({"pids": []}))
        self.assertEqual(config_pids({"pids": "test00"}), ["test00"])
        self.assertEqual(config_pids({"pids": "a, b"}), ["a", "b"])
        self.assertEqual(config_pids({"pids": 7}), ["7"])
        self.assertEqual(config_pids({"pids": ["a", 7]}), ["a", "7"])
        with self.assertRaises(TypeError):
            config_pids({"pids": {"a": 1}})


if __name__ == "__main__":
    unittest.main()

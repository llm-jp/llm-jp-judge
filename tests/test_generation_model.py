import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir


with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "src"), *sys.path]):
    evaluate = import_module("llm_jp_judge.evaluate")
    generate = import_module("llm_jp_judge.generate")
    remote = import_module("llm_jp_judge.client.remote")


class GenerationModelTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        dataset = root / "examples.json"
        dataset.write_text(json.dumps([{"ID": 1, "text": "Test question", "output": "Reference"}]), encoding="utf-8")
        output = root / "generation"
        self.responses = output / "safety_ja.jsonl"
        self.quality = output / "quality_ja.jsonl"
        self.metadata = output / "metadata.json"
        self.results = root / "evaluation"

        config_dir = str(Path(generate.__file__).parent / "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            self.gen = compose(config_name="generate", overrides=["client=openai"])
            self.eval = compose(config_name="evaluate", overrides=["client=openai"])
        self.gen.benchmark = {"safety_ja": self.gen.benchmark.safety_ja, "quality_ja": self.gen.benchmark.quality_ja}
        self.gen.benchmark.safety_ja.dataset.path = str(dataset)
        self.gen.output.dir = self.eval.input.dir = str(output)
        self.eval.output.dir = str(self.results)
        self.eval.client.model_name = "judge"

        self.generation_calls: list[str] = []
        sdk = patch.object(remote, "OpenAIClient")
        self.addCleanup(sdk.stop)
        sdk.start().return_value.chat.completions.create.side_effect = self.complete

    def complete(self, *, model, messages, **kwargs):
        if model == "judge":
            prompt = messages[-1]["content"]
            if "answer from model-A" in prompt:
                content = "[[5]]"
            else:
                self.assertIn("answer from model-B", prompt)
                content = "[[1]]"
        else:
            self.generation_calls.append(model)
            content = f"answer from {model}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def run_generation(self, model, *, overwrite=False):
        self.gen.client.model_name = model
        self.gen.output.overwrite = overwrite
        generate.main(self.gen)

    def assert_saved_result(self, model, score):
        response = json.loads(self.responses.read_text(encoding="utf-8"))
        self.assertEqual(response["response"], [f"answer from {model}"])
        self.assertEqual(json.loads(self.metadata.read_text(encoding="utf-8"))["model_name"], model)
        evaluate.main(self.eval)
        (row,) = json.loads((self.results / "score_table.json").read_text())
        self.assertEqual(row["generation_model"], model)
        self.assertEqual(row["safety_ja:安全性"], score)

    def test_same_model_reuses_responses(self):
        self.run_generation("model-A")
        self.run_generation("model-A")
        self.assertEqual(self.generation_calls, ["model-A"])
        self.assert_saved_result("model-A", 5)

    def test_different_model_without_overwrite_preserves_original_files(self):
        self.run_generation("model-A")
        responses_before = self.responses.read_bytes()
        metadata_before = self.metadata.read_bytes()
        with self.assertRaises(ValueError):
            self.run_generation("model-B")
        self.assertEqual(self.generation_calls, ["model-A"])
        self.assertEqual(self.responses.read_bytes(), responses_before)
        self.assertEqual(self.metadata.read_bytes(), metadata_before)
        self.assert_saved_result("model-A", 5)

    def test_different_model_with_overwrite_regenerates_responses(self):
        self.run_generation("model-A")
        self.run_generation("model-B", overwrite=True)
        self.assertEqual(self.generation_calls, ["model-A", "model-B"])
        self.assert_saved_result("model-B", 1)

    def test_legacy_responses_without_metadata_require_regeneration(self):
        self.run_generation("model-A")
        self.metadata.unlink()
        original = self.responses.read_bytes()
        with self.assertRaises(ValueError):
            self.run_generation("model-A")
        self.assertEqual(self.generation_calls, ["model-A"])
        self.assertFalse(self.metadata.exists())
        self.assertEqual(self.responses.read_bytes(), original)

        self.run_generation("model-B", overwrite=True)
        self.assertEqual(self.generation_calls, ["model-A", "model-B"])
        self.assert_saved_result("model-B", 1)

    def test_model_switch_rejects_unselected_responses(self):
        self.run_generation("model-A")
        leftover = self.responses.with_name("quality_ja.jsonl")
        original = self.responses.read_bytes()
        leftover.write_bytes(original)
        metadata_before = self.metadata.read_bytes()
        with self.assertRaises(ValueError):
            self.run_generation("model-B", overwrite=True)
        self.assertEqual(self.generation_calls, ["model-A"])
        self.assertEqual(self.responses.read_bytes(), original)
        self.assertEqual(leftover.read_bytes(), original)
        self.assertEqual(self.metadata.read_bytes(), metadata_before)

    def test_interrupted_model_switch_preserves_previous_answers_and_score(self):
        self.run_generation("model-A")
        responses_before = self.responses.read_bytes()
        metadata_before = self.metadata.read_bytes()
        generate_benchmark = generate.generate

        def interrupted(*args, **kwargs):
            generate_benchmark(*args, **kwargs)
            raise RuntimeError("interrupted after writing responses")

        with patch.object(generate, "generate", side_effect=interrupted), self.assertRaises(RuntimeError):
            self.run_generation("model-B", overwrite=True)
        self.assertEqual(self.generation_calls, ["model-A", "model-B"])
        self.assertEqual(self.responses.read_bytes(), responses_before)
        self.assertEqual(self.metadata.read_bytes(), metadata_before)
        self.assert_saved_result("model-A", 5)

    def test_interrupted_first_run_resumes_after_completed_benchmark(self):
        second_input = self.results.parent / "quality-examples.json"
        self.gen.benchmark.quality_ja.dataset.path = str(second_input)
        with self.assertRaises(FileNotFoundError):
            self.run_generation("model-A")
        completed = self.responses.read_bytes()
        self.assertEqual(json.loads(self.metadata.read_text(encoding="utf-8"))["model_name"], "model-A")
        self.assertFalse(self.quality.exists())
        self.assertEqual(self.generation_calls, ["model-A"])

        second_input.write_text('[{"ID": 2, "text": "Second question"}]', encoding="utf-8")
        self.run_generation("model-A")
        self.assertEqual(self.responses.read_bytes(), completed)
        self.assertEqual(json.loads(self.quality.read_text(encoding="utf-8"))["ID"], 2)
        self.assertEqual(self.generation_calls, ["model-A", "model-A"])

    def test_failed_second_replacement_restores_original_files_and_allows_retry(self):
        self.gen.benchmark.quality_ja.dataset.path = self.gen.benchmark.safety_ja.dataset.path
        self.run_generation("model-A")
        original = {p: p.read_bytes() for p in (self.responses, self.quality, self.metadata)}
        destinations = []
        replace = Path.replace

        def fail_second_response(source, target):
            target = Path(target)
            if target in (self.responses, self.quality) and source.parent != target.parent:
                if json.loads(source.read_bytes())["response"] == ["answer from model-B"]:
                    destinations.append(target)
                    if len(destinations) == 2:
                        raise OSError("second response replacement failed")
            return replace(source, target)

        with patch.object(Path, "replace", fail_second_response), self.assertRaisesRegex(OSError, "second response"):
            self.run_generation("model-B", overwrite=True)
        self.assertCountEqual(destinations, [self.responses, self.quality])
        self.assertEqual({p: p.read_bytes() for p in original}, original)

        self.run_generation("model-B", overwrite=True)
        self.assertEqual(json.loads(self.metadata.read_text(encoding="utf-8"))["model_name"], "model-B")
        for path in (self.responses, self.quality):
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["response"], ["answer from model-B"])

    def test_visible_answers_match_the_recorded_model_during_switch(self):
        self.gen.benchmark.quality_ja.dataset.path = self.gen.benchmark.safety_ja.dataset.path
        self.run_generation("model-A")
        output = self.metadata.parent
        snapshots = []
        replace = Path.replace

        def observe_replace(source, target):
            moved = replace(source, target)
            if output in (source.parent, moved.parent):
                model = json.loads(self.metadata.read_bytes())["model_name"]
                answers = [json.loads(path.read_bytes())["response"] for path in output.glob("*.jsonl")]
                snapshots.append((model, answers))
            return moved

        with patch.object(Path, "replace", observe_replace):
            self.run_generation("model-B", overwrite=True)

        for model, answers in snapshots:
            for response in answers:
                self.assertEqual(response, [f"answer from {model}"])
        self.assertEqual({model for model, answers in snapshots if len(answers) == 1}, {"model-A", "model-B"})

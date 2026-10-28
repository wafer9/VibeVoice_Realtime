"""Check requested Emilia score boundaries, ID joins, and transcript selection."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("emilia_meta", Path(__file__).resolve().parents[1] / "data/Emilia/meta.py")
META = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(META)


class EmiliaFilteringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def run_filter(self, language, rows, similarities, wers):
        metadata, sim, wer = [self.root / name for name in ["metadata.jsonl", "sim.list", "asr.wer"]]
        prompt = {**self.row(language, 999999), "text": "原始参考文本"}
        metadata.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*rows, prompt]), encoding="utf-8")
        sim.write_text("".join(f"{identifier}\t{prompt['id']}\t{score}\n" for identifier, score in similarities), encoding="utf-8")
        # The reference fails target WER criteria, but is still a valid lookup.
        wer.write_text("".join(f"{identifier}\t{score}\treference\t{text}\t0\t0\t0\n" for identifier, score, text in [*wers, (prompt['id'], .9, "Qwen prompt text")]), encoding="utf-8")
        output = io.BytesIO()
        report = META.filter_language(language, metadata, sim, wer, output)
        return [json.loads(line) for line in output.getvalue().splitlines()], report

    @staticmethod
    def row(language, number):
        return {"id": f"{language.upper()}_B00000_S00000_W{number:06d}", "text": "原始文本。",
                "duration": 6, "language": language, "wav": f"original/{number}.mp3", "dnsmos": 3.1}

    def test_strict_boundaries_and_english_hypothesis_joined_by_id(self):
        rows = [self.row("en", i) for i in range(5)]
        ids = [row["id"] for row in rows]
        result, report = self.run_filter("en", rows, [(ids[i], score) for i, score in [(4, .9), (1, .9), (0, .8), (3, .81), (2, .9)]],
                                         [(ids[i], score, f"Qwen text {i}!") for i, score in [(3, .0999), (2, .10001), (1, .1), (0, 0), (4, 0)]])
        self.assertEqual([row["id"] for row in result], [ids[3], ids[4]])
        self.assertEqual(result[0], {**rows[3], "text": "Qwen text 3!",
                                     "prompt_wav": "EN_B00000_S00000_W999999", "prompt_text": "Qwen prompt text"})
        self.assertEqual(report["counts"]["accepted"], 2)
        self.assertAlmostEqual(report["accepted_hours"], 12 / 3600)

    def test_chinese_keeps_original_metadata_and_missing_scores_are_excluded(self):
        rows = [self.row("zh", i) for i in range(4)]
        ids = [row["id"] for row in rows]
        result, report = self.run_filter("zh", rows, [(ids[0], .9), (ids[1], .9), (ids[3], .9)],
                                         [(ids[0], .09, "不要使用这个中文识别结果"), (ids[1], .1, "bad"), (ids[2], 0, "no sim")])
        self.assertEqual(result, [{**rows[0], "prompt_wav": "ZH_B00000_S00000_W999999", "prompt_text": "原始参考文本"}])
        self.assertEqual(report["counts"]["metadata_missing_wer"], 1)
        self.assertEqual(report["counts"]["metadata_without_passing_sim"], 2)

    def test_nonfinite_scores_never_pass(self):
        rows = [self.row("en", i) for i in range(3)]
        ids = [row["id"] for row in rows]
        result, report = self.run_filter("en", rows, [(ids[0], "nan"), (ids[1], .9), (ids[2], .9)],
                                         [(ids[0], 0, "x"), (ids[1], "nan", "y"), (ids[2], "-inf", "z")])
        self.assertEqual(result, [])
        self.assertEqual(report["counts"]["sim_invalid_score"], 1)
        self.assertEqual(report["counts"]["wer_invalid_score"], 2)

    def test_duplicate_joined_id_fails_instead_of_choosing_silently(self):
        row = self.row("en", 0)
        with self.assertRaisesRegex(ValueError, "duplicate joined metadata ID"):
            self.run_filter("en", [row, row], [(row["id"], .9)], [(row["id"], 0, "text")])


if __name__ == "__main__":
    unittest.main()

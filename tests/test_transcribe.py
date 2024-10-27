import unittest
from transcribe import _process_segments, TranscribeResult


class TestProcessSegments(unittest.TestCase):
    def setUp(self):
        self.token_spans = [
            type("TokenSpan", (object,), {"token": t, "start": i, "end": i + 1})
            for i, t in enumerate(
                [
                    # Mock token spans with token, start time, and end time
                    # Assuming token IDs are integers and start/end times are in seconds
                    17498,
                    10153,
                    12616,
                    19768,
                    24879,
                    12382,
                    10153,
                    11754,
                    20046,
                ]
            )
        ]
        self.ratio = 1.0
        self.with_punct = False
        self.punct_labels = "?!。，；？！"  # corresponding token IDs [9705, 9676, 9729, 24879, 24882, 24883, 20046]

        # Mocking asr_model and special_token_ids
        global asr_model, special_token_ids
        asr_model = type(
            "ASRModel",
            (object,),
            {
                "tokenizer": type(
                    "Tokenizer",
                    (object,),
                    {
                        "sp": type(
                            "SP",
                            (object,),
                            {"IdToPiece": lambda self, token: f"token_{token}"},
                        )()
                    },
                )()
            },
        )()
        special_token_ids = [3, 4]  # Assuming tokens 3 and 4 are special tokens

    def test_process_segments_without_punctuation(self):
        results = _process_segments(
            self.token_spans, self.ratio, self.with_punct, self.punct_labels
        )
        expected_results = [
            TranscribeResult(text="話你戇鳩 怕你嬲", start_time=0, end_time=9.0),
        ]
        self.assertEqual(len(results), len(expected_results))
        for result, expected in zip(results, expected_results):
            self.assertEqual(result.text, expected.text)
            self.assertEqual(result.start_time, expected.start_time)
            self.assertEqual(result.end_time, expected.end_time)

    def test_process_segments_with_punctuation(self):
        self.with_punct = True
        self.token_spans.append(
            type("TokenSpan", (object,), {"token": 5, "start": 2.0, "end": 2.5})
        )
        results = _process_segments(
            self.token_spans, self.ratio, self.with_punct, self.punct_labels
        )
        expected_results = [
            TranscribeResult(text="話你戇鳩，怕你嬲！", start_time=0, end_time=9.0),
        ]
        self.assertEqual(len(results), len(expected_results))
        for result, expected in zip(results, expected_results):
            self.assertEqual(result.text, expected.text)
            self.assertEqual(result.start_time, expected.start_time)
            self.assertEqual(result.end_time, expected.end_time)


if __name__ == "__main__":
    unittest.main()

    # python -m unittest tests/test_transcribe.py

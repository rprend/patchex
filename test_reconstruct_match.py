import unittest

from text2fx_gemini.reconstruct_match import sanitize_session


class SanitizeSessionTest(unittest.TestCase):
    def test_accepts_return_send_map(self) -> None:
        session = sanitize_session(
            {
                "layers": [
                    {
                        "synth": {"note": 48},
                        "effects": {"return_send": {"space": 0.08}},
                    }
                ],
                "returns": [{"id": "space", "type": "reverb"}],
            },
            duration=5.0,
            sample_rate=44100,
        )

        self.assertEqual(session["layers"][0]["effects"]["return_send"], 0.08)

    def test_ignores_invalid_return_send_map_values(self) -> None:
        session = sanitize_session(
            {
                "layers": [
                    {
                        "synth": {"note": 48},
                        "effects": {"return_send": {"space": "loud"}},
                    }
                ],
            },
            duration=5.0,
            sample_rate=44100,
        )

        self.assertEqual(session["layers"][0]["effects"]["return_send"], 0.0)


if __name__ == "__main__":
    unittest.main()

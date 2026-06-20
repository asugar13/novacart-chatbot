import unittest
from unittest.mock import patch

from escalation import assess_escalation


class EscalationTests(unittest.TestCase):
    def test_explicit_human_request_always_offers_handoff(self):
        result = assess_escalation(
            "Please connect me to a human agent.",
            mode="hybrid",
        )

        self.assertTrue(result.should_offer)
        self.assertTrue(result.explicit_request)
        self.assertEqual(result.detector, "rules")

    @patch(
        "escalation.get_emotion_scores",
        return_value={"anger": 0.08, "disgust": 0.03, "sadness": 0.65},
    )
    def test_negative_event_is_not_automatically_escalated(self, _mock_scores):
        result = assess_escalation(
            "My package is delayed.",
            mode="hybrid",
        )

        self.assertFalse(result.should_offer)
        self.assertEqual(result.streak, 0)

    @patch(
        "escalation.get_emotion_scores",
        return_value={"anger": 0.86, "disgust": 0.05, "neutral": 0.04},
    )
    def test_high_anger_offers_handoff(self, _mock_scores):
        result = assess_escalation(
            "Why has nobody fixed this?",
            mode="hybrid",
        )

        self.assertTrue(result.should_offer)
        self.assertEqual(result.detector, "emotion")

    @patch(
        "escalation.get_emotion_scores",
        return_value={"anger": 0.48, "disgust": 0.08, "neutral": 0.30},
    )
    def test_repeated_medium_frustration_escalates_on_second_turn(self, _mock_scores):
        result = assess_escalation(
            "It still has not been fixed.",
            mode="emotion",
            previous_streak=1,
        )

        self.assertTrue(result.should_offer)
        self.assertEqual(result.detector, "trend")
        self.assertEqual(result.streak, 2)

    def test_strong_frustration_phrase_uses_fast_rule(self):
        result = assess_escalation(
            "This service is absolutely unacceptable.",
            mode="hybrid",
        )

        self.assertTrue(result.should_offer)
        self.assertEqual(result.detector, "rules")

    @patch(
        "escalation.classify_with_qwen",
        return_value={
            "frustration": "high",
            "offer_handoff": True,
            "confidence": 0.91,
            "reason": "Repeated unresolved support failure",
        },
    )
    def test_qwen_mode_uses_context_aware_classifier(self, mock_qwen):
        result = assess_escalation(
            "Nothing has changed.",
            ["I reported the missing parcel yesterday."],
            mode="qwen",
        )

        self.assertTrue(result.should_offer)
        self.assertEqual(result.detector, "qwen")
        mock_qwen.assert_called_once()


if __name__ == "__main__":
    unittest.main()

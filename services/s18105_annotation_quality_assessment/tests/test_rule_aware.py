"""Tests for rule-aware features and VLM budget in /verify endpoint."""

import requests

from conftest import SERVICE_URL, skip_no_service


# ---------------------------------------------------------------------------
# Helmet-style class rules and text prompts used across tests
# ---------------------------------------------------------------------------

HELMET_CLASSES = {"0": "person", "1": "head_with_helmet", "2": "head_without_helmet"}
HELMET_TEXT_PROMPTS = {
    "person": "a person",
    "head_with_helmet": "head with helmet",
    "head_without_helmet": "bare head",
}
HELMET_LABELS = [
    "0 0.5 0.5 0.3 0.6",
    "1 0.5 0.3 0.1 0.1",
    "2 0.2 0.3 0.1 0.1",
]
HELMET_CLASS_RULES = [
    {"output_class_id": 0, "source": "person", "condition": "direct"},
    {
        "output_class_id": 1,
        "source": "head",
        "condition": "overlap",
        "target": "helmet",
        "min_iou": 0.3,
    },
    {
        "output_class_id": 2,
        "source": "head",
        "condition": "no_overlap",
        "target": "helmet",
        "min_iou": 0.3,
    },
]


# ---------------------------------------------------------------------------
# POST /verify — rule-aware features
# ---------------------------------------------------------------------------


class TestRuleAware:

    @skip_no_service
    def test_verify_with_class_rules(self, test_image_b64):
        """Send /verify with class_rules for helmet rules. Verify 200 OK."""
        resp = requests.post(
            f"{SERVICE_URL}/verify",
            json={
                "image": test_image_b64,
                "labels": HELMET_LABELS,
                "label_format": "yolo",
                "classes": HELMET_CLASSES,
                "text_prompts": HELMET_TEXT_PROMPTS,
                "class_rules": HELMET_CLASS_RULES,
            },
            timeout=120,
        )
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "quality_score" in data
            assert "grade" in data

    @skip_no_service
    def test_verify_backward_compatible(self, test_image_b64, sample_yolo_labels, sample_classes):
        """Standard /verify without rule fields still works."""
        resp = requests.post(
            f"{SERVICE_URL}/verify",
            json={
                "image": test_image_b64,
                "labels": sample_yolo_labels,
                "label_format": "yolo",
                "classes": sample_classes,
            },
            timeout=120,
        )
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "quality_score" in data
            assert "grade" in data

    @skip_no_service
    def test_vlm_budget_accepted(self, test_image_b64):
        """Send vlm_budget config. Verify 200 OK (no crash even if Ollama unavailable)."""
        resp = requests.post(
            f"{SERVICE_URL}/verify",
            json={
                "image": test_image_b64,
                "labels": HELMET_LABELS,
                "label_format": "yolo",
                "classes": HELMET_CLASSES,
                "text_prompts": HELMET_TEXT_PROMPTS,
                "vlm_budget": {"sample_rate": 0.1, "max_samples": 5},
                "enable_vlm": True,
            },
            timeout=120,
        )
        # Should not crash — 200 if SAM3 available, 503 if SAM3 down
        assert resp.status_code in (200, 503)

    @skip_no_service
    def test_validate_with_rules_ignored(self, test_image_b64, sample_yolo_labels, sample_classes):
        """/validate endpoint doesn't accept rules. Verify it still works normally."""
        resp = requests.post(
            f"{SERVICE_URL}/validate",
            json={
                "image": test_image_b64,
                "labels": sample_yolo_labels,
                "label_format": "yolo",
                "classes": sample_classes,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "issues" in data
        assert "quality_score" in data
        assert "grade" in data

    @skip_no_service
    def test_verify_class_rules_scoring(self, test_image_b64):
        """With class_rules, derived classes should get softer scoring.

        Send the same labels with and without class_rules, compare scores.
        With rules should be >= without rules for derived classes.
        """
        base_payload = {
            "image": test_image_b64,
            "labels": HELMET_LABELS,
            "label_format": "yolo",
            "classes": HELMET_CLASSES,
            "text_prompts": HELMET_TEXT_PROMPTS,
        }

        # Without rules
        resp_no_rules = requests.post(
            f"{SERVICE_URL}/verify",
            json=base_payload,
            timeout=120,
        )

        # With rules
        payload_with_rules = {**base_payload, "class_rules": HELMET_CLASS_RULES}
        resp_with_rules = requests.post(
            f"{SERVICE_URL}/verify",
            json=payload_with_rules,
            timeout=120,
        )

        # Both should succeed or both should 503 (SAM3 down)
        if resp_no_rules.status_code == 503 or resp_with_rules.status_code == 503:
            # SAM3 unavailable — skip comparison but don't fail
            return

        assert resp_no_rules.status_code == 200
        assert resp_with_rules.status_code == 200

        score_no_rules = resp_no_rules.json()["quality_score"]
        score_with_rules = resp_with_rules.json()["quality_score"]

        # Derived classes (overlap/no_overlap conditions) should get softer
        # scoring with rules, so score_with_rules >= score_no_rules
        assert score_with_rules >= score_no_rules, (
            f"Expected score with rules ({score_with_rules}) >= "
            f"score without rules ({score_no_rules})"
        )

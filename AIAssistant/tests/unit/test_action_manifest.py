import os
import sys
import unittest

# Self-bootstrap: the eval CI job runs this file directly from the repo root,
# where neither the repo layout nor tests/__init__.py puts ``src`` on sys.path
# (the standalone-run pattern used by evals/run_agentic_eval.py).
_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from ai_assistant.tools.action_manifest import (
    action_ids,
    action_intents,
    canonical_action,
    manifest,
    rank_actions_for_step,
    step_matches_action,
)


class ManifestContractTests(unittest.TestCase):
    def test_every_action_has_description_and_intents(self):
        for entry in manifest()["actions"]:
            self.assertTrue(entry.get("description"), entry["id"])
            self.assertTrue(entry.get("intents"), entry["id"])

    def test_own_intents_never_lose_to_another_action(self):
        # Catches ambiguous phrases at CI time instead of in production.
        for action_id in action_ids():
            for phrase in action_intents(action_id):
                ranked = dict(rank_actions_for_step(phrase))
                self.assertEqual(ranked.get(action_id), max(ranked.values()), f"{action_id}: {phrase} -> {ranked}")

    def test_hide_step_maps_to_close_not_view(self):
        self.assertEqual(rank_actions_for_step("Ẩn mô hình 3D")[0][0], "reconstruction.close_3d_model")
        self.assertIs(step_matches_action("Ẩn mô hình 3D", "reconstruction.view_3d_model"), False)
        self.assertIs(step_matches_action("Ẩn mô hình 3D", "reconstruction.close_3d_model"), True)

    def test_unaccented_and_english_steps(self):
        self.assertIs(step_matches_action("an mo hinh 3d", "reconstruction.close_3d_model"), True)
        self.assertIs(step_matches_action("Hide the 3D model", "reconstruction.view_3d_model"), False)

    def test_steps_without_evidence_are_not_blocked(self):
        self.assertIsNone(step_matches_action("Đọc file README", "viewer.load_2d"))

    def test_alias_resolves_to_canonical(self):
        self.assertEqual(canonical_action("mail.inbox"), "mail.open")


if __name__ == "__main__":
    unittest.main()

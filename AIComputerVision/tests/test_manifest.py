"""Unit tests for the model manifest (integrity + provenance)."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvtrain import manifest as manifest_mod
from cvtrain.config import MANIFEST_NAME, MODEL_SPECS
from cvtrain.fsio import sha256_file


class Sha256Tests(unittest.TestCase):
    def test_matches_reference_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            payload = b"hello world"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(sha256_file(path), expected)


class ManifestRoundTripTests(unittest.TestCase):
    def test_missing_manifest_returns_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = manifest_mod.load_manifest(Path(tmp))
            self.assertEqual(data["entries"], {})
            self.assertEqual(data["schema_version"], manifest_mod.SCHEMA_VERSION)

    def test_corrupt_manifest_returns_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
            with self.assertLogs("cvtrain.manifest", level="WARNING") as captured:
                data = manifest_mod.load_manifest(Path(tmp))
            self.assertEqual(data["entries"], {})
            # Corruption must be reported, not silently swallowed.
            self.assertIn("Corrupt manifest", captured.output[0])

    def test_save_is_atomic_and_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            skeleton: manifest_mod.Manifest = {"schema_version": 1, "entries": {}}
            manifest_mod.upsert_entry(skeleton, "yolo11n.onnx", {"sha256": "abc"})
            self.assertEqual(skeleton["entries"]["yolo11n.onnx"]["sha256"], "abc")

            data: manifest_mod.Manifest = {
                "schema_version": 1,
                "entries": {"yolo11n.onnx": {"sha256": "abc"}},
            }
            path = manifest_mod.save_manifest(models_dir, data)
            self.assertTrue(path.exists())
            self.assertEqual(manifest_mod.load_manifest(models_dir)["entries"]["yolo11n.onnx"]["sha256"], "abc")
            leftovers = [p.name for p in models_dir.iterdir() if p.name.startswith(".manifest.")]
            self.assertEqual(leftovers, [])

    def test_verify_detects_ok_missing_and_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            good = models_dir / "yolo11n.onnx"
            good.write_bytes(b"model-bytes")
            manifest: manifest_mod.Manifest = {
                "schema_version": 1,
                "entries": {
                    "yolo11n.onnx": {"sha256": sha256_file(good)},
                    "ghost.onnx": {"sha256": "deadbeef"},
                },
            }
            problems = manifest_mod.verify_manifest(models_dir, manifest)
            self.assertEqual(problems, ["ghost.onnx: missing on disk"])

            good.write_bytes(b"tampered")
            problems = manifest_mod.verify_manifest(models_dir, manifest)
            self.assertTrue(any("sha256 mismatch" in problem for problem in problems))

    def test_verify_detects_a_file_present_without_an_entry(self) -> None:
        # Bidirectional: a stray ONNX dropped into Models/ with no manifest entry
        # must be reported, otherwise the manifest stops accounting for it.
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo11n.onnx").write_bytes(b"stray-model-bytes")
            manifest: manifest_mod.Manifest = {"schema_version": 1, "entries": {}}
            self.assertEqual(
                manifest_mod.verify_manifest(models_dir, manifest),
                ["yolo11n.onnx: on disk but not recorded in the manifest"],
            )

    def test_verify_ignores_non_onnx_files_and_tracked_temp_files(self) -> None:
        # Only published artifacts (``*.onnx``) count as models; an in-progress
        # atomic write's temp file must not be reported as an orphan.
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            good = models_dir / "yolo11n.onnx"
            good.write_bytes(b"model-bytes")
            (models_dir / "out.onnx.tmp").write_bytes(b"partial")
            (models_dir / "notes.txt").write_text("not a model", encoding="utf-8")
            manifest: manifest_mod.Manifest = {
                "schema_version": 1,
                "entries": {"yolo11n.onnx": {"sha256": sha256_file(good)}},
            }
            self.assertEqual(manifest_mod.verify_manifest(models_dir, manifest), [])

    def test_verify_ignores_the_rollback_backup(self) -> None:
        # ``yolo11n.onnx.bak`` is a recovery artifact, not a model: it has no
        # manifest entry by design and must not be reported as untracked.
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            good = models_dir / "yolo11n.onnx"
            good.write_bytes(b"model-bytes")
            (models_dir / "yolo11n.onnx.bak").write_bytes(b"older-bytes")
            manifest: manifest_mod.Manifest = {
                "schema_version": 1,
                "entries": {"yolo11n.onnx": {"sha256": sha256_file(good)}},
            }
            self.assertEqual(manifest_mod.verify_manifest(models_dir, manifest), [])

    def test_build_from_scan_records_only_present_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / MODEL_SPECS["det"].output).write_bytes(b"det")
            manifest = manifest_mod.build_from_scan(models_dir)
            self.assertIn(MODEL_SPECS["det"].output, manifest["entries"])
            self.assertNotIn(MODEL_SPECS["seg"].output, manifest["entries"])
            entry = manifest["entries"][MODEL_SPECS["det"].output]
            self.assertEqual(entry["role"], "detection")
            self.assertEqual(entry["size_bytes"], 3)

    def test_build_from_scan_keeps_metadata_when_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            target = models_dir / MODEL_SPECS["det"].output
            target.write_bytes(b"det")
            manifest = manifest_mod.build_from_scan(models_dir)
            manifest["entries"][MODEL_SPECS["det"].output]["trained_at"] = "2026-01-01T00:00:00+00:00"
            manifest_mod.save_manifest(models_dir, manifest)

            refreshed = manifest_mod.build_from_scan(models_dir)
            entry = refreshed["entries"][MODEL_SPECS["det"].output]
            self.assertEqual(entry["trained_at"], "2026-01-01T00:00:00+00:00")

    def test_build_from_scan_drops_stale_provenance_after_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            target = models_dir / MODEL_SPECS["det"].output
            target.write_bytes(b"original")
            manifest = manifest_mod.build_from_scan(models_dir)
            entry = manifest["entries"][MODEL_SPECS["det"].output]
            entry["trained_at"] = "2026-01-01T00:00:00+00:00"
            entry["metrics"] = {"metrics/mAP50(B)": 0.9}
            manifest_mod.save_manifest(models_dir, manifest)

            target.write_bytes(b"replaced")
            refreshed = manifest_mod.build_from_scan(models_dir)
            entry = refreshed["entries"][MODEL_SPECS["det"].output]
            self.assertIsNone(entry["trained_at"])
            self.assertNotIn("metrics", entry)
            self.assertEqual(entry["sha256"], sha256_file(target))

    def test_scan_handles_a_changed_file_whose_identity_fields_are_absent(self) -> None:
        # An entry written by an older schema may carry only a digest; the scan
        # must still drop provenance and re-derive identity from the spec.
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            target = models_dir / MODEL_SPECS["det"].output
            target.write_bytes(b"original")
            manifest = manifest_mod.build_from_scan(models_dir)
            entry = manifest["entries"][MODEL_SPECS["det"].output]
            entry.pop("role", None)
            entry.pop("source_weights", None)
            manifest_mod.save_manifest(models_dir, manifest)

            target.write_bytes(b"replaced")
            refreshed = manifest_mod.build_from_scan(models_dir)
            entry = refreshed["entries"][MODEL_SPECS["det"].output]
            self.assertEqual(entry["role"], MODEL_SPECS["det"].role)
            self.assertEqual(entry["source_weights"], MODEL_SPECS["det"].ckpt)
            self.assertIsNone(entry["trained_at"])


class SchemaRobustnessTests(unittest.TestCase):
    """A malformed document must degrade to an empty manifest, never crash."""

    def test_entries_of_the_wrong_type_falls_back_to_a_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / MANIFEST_NAME
            path.write_text('{"entries": ["not", "a", "map"]}', encoding="utf-8")
            with self.assertLogs("cvtrain.manifest", level="WARNING") as captured:
                data = manifest_mod.load_manifest(Path(tmp))
            self.assertEqual(data["entries"], {})
            self.assertIn("invalid", captured.output[0])

    def test_missing_entries_key_falls_back_to_a_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / MANIFEST_NAME).write_text('{"schema_version": 1}', encoding="utf-8")
            with self.assertLogs("cvtrain.manifest", level="WARNING"):
                data = manifest_mod.load_manifest(Path(tmp))
            self.assertEqual(data["entries"], {})
            self.assertEqual(data["schema_version"], manifest_mod.SCHEMA_VERSION)

    def test_unreadable_manifest_falls_back_to_a_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # A directory in place of the file makes read_text raise OSError.
            (Path(tmp) / MANIFEST_NAME).mkdir()
            with self.assertLogs("cvtrain.manifest", level="WARNING"):
                data = manifest_mod.load_manifest(Path(tmp))
            self.assertEqual(data["entries"], {})

    def test_upsert_creates_the_entries_map_when_absent(self) -> None:
        manifest: manifest_mod.Manifest = {"schema_version": 1}
        manifest_mod.upsert_entry(manifest, "a.onnx", {"role": "detection"})
        self.assertEqual(manifest["entries"]["a.onnx"]["role"], "detection")

    def test_verify_flags_entries_without_a_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo11n.onnx").write_bytes(b"bytes")
            manifest: manifest_mod.Manifest = {"entries": {"yolo11n.onnx": {"role": "detection"}}}
            self.assertEqual(
                manifest_mod.verify_manifest(models_dir, manifest),
                ["yolo11n.onnx: entry has no sha256"],
            )

    def test_verify_ignores_entries_that_are_not_on_disk_yet(self) -> None:
        # build_from_scan may reference a file that a concurrent run removed.
        manifest: manifest_mod.Manifest = {"entries": {"ghost.onnx": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                manifest_mod.verify_manifest(Path(tmp), manifest),
                ["ghost.onnx: missing on disk"],
            )


class SchemaMigrationTests(unittest.TestCase):
    """``schema_version`` is a contract: read it, upgrade it, never down-stamp."""

    def _load(self, payload: str) -> manifest_mod.Manifest:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / MANIFEST_NAME
            path.write_text(payload, encoding="utf-8")
            return manifest_mod.load_manifest(Path(tmp))

    def test_unversioned_document_is_upgraded_to_the_current_schema(self) -> None:
        # Schema 0: written before the field existed. Entries must survive the
        # upgrade untouched — only the header is rewritten.
        manifest = self._load('{"entries": {"yolo11n.onnx": {"sha256": "abc"}}}')
        self.assertEqual(manifest["schema_version"], manifest_mod.SCHEMA_VERSION)
        self.assertIn("cvtrain", manifest["generator"])
        self.assertEqual(manifest["entries"]["yolo11n.onnx"]["sha256"], "abc")

    def test_a_current_document_keeps_its_own_header(self) -> None:
        manifest = self._load('{"schema_version": 1, "generator": "cvtrain 0.9.0", "entries": {}}')
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["generator"], "cvtrain 0.9.0")

    def test_a_document_from_a_newer_build_is_read_as_is(self) -> None:
        # Down-stamping would erase the signal a future reader needs in order to
        # run its own migration, so the version is reported and left alone.
        with self.assertLogs("cvtrain.manifest", level="WARNING") as captured:
            manifest = self._load('{"schema_version": 99, "entries": {}}')
        self.assertEqual(manifest["schema_version"], 99)
        self.assertTrue(any("not down-stamping" in line for line in captured.output), captured.output)

    def test_a_non_numeric_version_is_treated_as_unversioned(self) -> None:
        manifest = self._load('{"schema_version": "1", "entries": {}}')
        self.assertEqual(manifest["schema_version"], manifest_mod.SCHEMA_VERSION)

    def test_a_negative_version_is_treated_as_unversioned(self) -> None:
        # A corrupt header must not make the migration chain index a negative key.
        manifest = self._load('{"schema_version": -3, "entries": {}}')
        self.assertEqual(manifest["schema_version"], manifest_mod.SCHEMA_VERSION)

    def test_a_disk_scan_does_not_down_stamp_a_newer_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / MODEL_SPECS["det"].output).write_bytes(b"det")
            (models_dir / MANIFEST_NAME).write_text('{"schema_version": 99, "entries": {}}', encoding="utf-8")

            with self.assertLogs("cvtrain.manifest", level="WARNING"):
                manifest = manifest_mod.build_from_scan(models_dir)

            self.assertEqual(manifest["schema_version"], 99)
            self.assertIn(MODEL_SPECS["det"].output, manifest["entries"])


class ManifestQuarantineTests(unittest.TestCase):
    """A corrupt manifest is preserved, never silently overwritten."""

    def test_a_corrupt_manifest_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            path = models_dir / MANIFEST_NAME
            path.write_text("{not json", encoding="utf-8")

            with self.assertLogs("cvtrain.manifest", level="WARNING"):
                manifest = manifest_mod.load_manifest(models_dir)

            self.assertEqual(manifest["entries"], {})
            self.assertFalse(path.exists())
            quarantined = list(models_dir.glob(f"{MANIFEST_NAME}.corrupt-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_text(encoding="utf-8"), "{not json")

    def test_an_invalid_manifest_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / MANIFEST_NAME).write_text('{"entries": []}', encoding="utf-8")

            with self.assertLogs("cvtrain.manifest", level="WARNING"):
                manifest = manifest_mod.load_manifest(models_dir)

            self.assertEqual(manifest["entries"], {})
            self.assertEqual(len(list(models_dir.glob(f"{MANIFEST_NAME}.corrupt-*"))), 1)

    def test_a_failed_quarantine_still_returns_the_skeleton(self) -> None:
        # A read-only volume must not stop the run: the corruption is logged and
        # the reader still fails closed to an empty manifest.
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

            with mock.patch("cvtrain.manifest.os.replace", side_effect=OSError("read-only")):
                with self.assertLogs("cvtrain.manifest", level="WARNING") as captured:
                    manifest = manifest_mod.load_manifest(models_dir)

            self.assertEqual(manifest["entries"], {})
            self.assertTrue(any("Could not quarantine" in line for line in captured.output))


class ScopedVerificationTests(unittest.TestCase):
    """``verify_manifest(only=...)`` checks the named artifacts and no more."""

    def test_only_ignores_an_unrelated_entry_and_a_stray_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            good = models_dir / "yolo11n.onnx"
            good.write_bytes(b"model-bytes")
            other = models_dir / "other.onnx"
            other.write_bytes(b"other-bytes")
            (models_dir / "stray.onnx").write_bytes(b"stray-bytes")
            manifest: manifest_mod.Manifest = {
                "schema_version": 1,
                "entries": {
                    "yolo11n.onnx": {"sha256": sha256_file(good)},
                    "other.onnx": {"sha256": sha256_file(other)},
                },
            }

            self.assertEqual(manifest_mod.verify_manifest(models_dir, manifest, only=["yolo11n.onnx"]), [])
            # The global sweep is unchanged: it still accounts for the stray file.
            self.assertEqual(
                manifest_mod.verify_manifest(models_dir, manifest),
                ["stray.onnx: on disk but not recorded in the manifest"],
            )

    def test_only_still_reports_a_named_artifact_without_an_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest: manifest_mod.Manifest = {"entries": {"yolo11n.onnx": {"sha256": "deadbeef"}}}
            self.assertEqual(
                manifest_mod.verify_manifest(Path(tmp), manifest, only=["yolo11n.onnx", "seg.onnx"]),
                ["yolo11n.onnx: missing on disk", "seg.onnx: expected but not recorded in the manifest"],
            )


if __name__ == "__main__":
    unittest.main()

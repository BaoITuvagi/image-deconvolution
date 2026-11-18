"""Reproducibility and six saved-reference regression checks."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_deconvolution.experiments.export import (
    MissingInput,
    export_bundle,
    load_bundle,
    load_experiment,
)
from image_deconvolution.experiments.reference import (
    compare_references,
    research_example,
    verify_resources,
)
from image_deconvolution.experiments.runs import RunConfig, restore
from image_deconvolution.io.images import read_image


class ExportReferenceTests(unittest.TestCase):
    def test_six_saved_reference_comparisons(self):
        verify_resources()
        results = compare_references()
        self.assertEqual(len(results), 6)
        for result in results:
            self.assertLess(
                result["max_abs"], 1e-6, (result["image"], result["method"])
            )
            self.assertLess(result["rmse"], 1e-6)
            self.assertAlmostEqual(
                result["reference_quality_python"]["psnr"],
                result["saved_imagej_psnr"],
                places=5,
            )

    def test_bundle_round_trip_csv_and_output_values(self):
        observation, psf, truth = research_example("cells")
        record = restore(observation, psf, RunConfig(5, inspect=True), truth)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = Path(directory) / "experiment with spaces"
            path = export_bundle(folder, record)
            with self.assertRaises(FileExistsError):
                export_bundle(folder, record)
            images, config, degradation = load_bundle(path)
            self.assertEqual(config, record.config)
            self.assertIsNone(degradation)
            np.testing.assert_array_equal(
                images["observation"].pixels, record.observation.pixels
            )
            rerun = restore(
                images["observation"], images["psf"], config, images["ground_truth"]
            )
            np.testing.assert_allclose(rerun.output, record.output, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(
                read_image(folder / "RL_output.tif").pixels,
                record.output.astype(np.float32),
            )
            with (folder / "measurements.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            for key in ["mse", "psnr", "ssim"]:
                self.assertEqual(float(row[key]), record.quality[key])
                self.assertEqual(float(row[f"input_{key}"]), record.input_quality[key])
            self.assertEqual(float(row["psnr_change_db"]), record.psnr_change)
            data = json.loads(path.read_text())
            self.assertEqual(data["conventions"]["boundaries"], "circular")
            self.assertIn("numpy", data["environment"])
            self.assertEqual(data["input_measurements"], record.input_quality)
            saved = load_experiment(path)
            self.assertEqual(saved.config, record.config)
            self.assertEqual(saved.seconds, record.seconds)
            self.assertEqual(saved.quality, record.quality)
            self.assertEqual(saved.input_quality, record.input_quality)
            self.assertEqual(saved.curve, record.curve)
            self.assertEqual(saved.saved_from, str(path.resolve()))
            np.testing.assert_array_equal(saved.output, record.output)
            self.assertFalse(saved.output.flags.writeable)
            self.assertEqual(list(saved.steps), list(record.steps))
            for iteration, phases in saved.steps.items():
                self.assertEqual(list(phases), list(record.steps[iteration]))
                for phase, array in phases.items():
                    np.testing.assert_array_equal(array, record.steps[iteration][phase])
                    self.assertFalse(array.flags.writeable)

    def test_saved_methods_without_capture_or_ground_truth(self):
        observation, psf, _ = research_example("bars")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            for method, capture in [
                ("RL", False),
                ("RL", True),
                ("IF", True),
                ("TRIF", True),
            ]:
                with self.subTest(method=method, capture=capture):
                    record = restore(
                        observation, psf, RunConfig(5, method=method, inspect=capture)
                    )
                    path = export_bundle(
                        Path(directory) / f"{method}_{capture}", record
                    )
                    saved = load_experiment(path)
                    np.testing.assert_array_equal(saved.output, record.output)
                    self.assertIsNone(saved.quality)
                    self.assertIsNone(saved.input_quality)
                    self.assertEqual(saved.curve, [])
                    self.assertEqual(bool(saved.steps), method == "RL" and capture)
                    self.assertEqual(
                        bool(saved.inspection_note), method == "RL" and not capture
                    )

    def test_legacy_reopens_tiff_recorded_measurements_and_curve_without_steps(self):
        observation, psf, truth = research_example("cells")
        record = restore(observation, psf, RunConfig(5, inspect=True), truth)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = export_bundle(Path(directory) / "legacy", record)
            data = json.loads(path.read_text())
            data["schema_version"] = 1
            del data["result"]
            path.write_text(json.dumps(data))
            saved = load_experiment(path)
            np.testing.assert_array_equal(
                saved.output, record.output.astype(np.float32)
            )
            self.assertEqual(saved.quality, record.quality)
            self.assertEqual(saved.seconds, record.seconds)
            self.assertEqual(saved.curve, record.curve)
            self.assertEqual(saved.steps, {})
            self.assertIn("legacy export", saved.inspection_note)
            self.assertIn("rounding", saved.metric_note)

    def test_invalid_saved_results_rejected_before_loading(self):
        observation, psf, truth = research_example("cells")
        record = restore(observation, psf, RunConfig(5, inspect=True), truth)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = export_bundle(Path(directory) / "saved", record)
            original = json.loads(path.read_text())
            cases = [
                ("result.array.file", "../output.npy"),
                ("result.array.file_sha256", "changed"),
                ("result.array.array_sha256", "changed"),
                ("result.algorithm_seconds", -1),
                ("result.checkpoints", original["result"]["checkpoints"] * 2),
                ("result.curve", original["result"]["curve"] * 7),
                ("config.inspect", False),
                ("measurements.ssim", 2),
                ("input_measurements.mse", -1),
                ("inputs.ground_truth", None),
            ]
            for key, value in cases:
                with self.subTest(key=key):
                    data = copy.deepcopy(original)
                    target = data
                    parts = key.split(".")
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaises((ValueError, TypeError)):
                        load_experiment(path)
            data = copy.deepcopy(original)
            del data["result"]["checkpoints"][0]["phases"]["Estimate x(k)"]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "five phases"):
                load_experiment(path)
            path.write_text(json.dumps(original))
            array_file = path.parent / "output.npy"
            np.save(array_file, np.zeros((12, 12)))
            original["result"]["array"]["file_sha256"] = hashlib.sha256(
                array_file.read_bytes()
            ).hexdigest()
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_experiment(path)
            array_file.unlink()
            with self.assertRaisesRegex(ValueError, "Missing saved result"):
                load_experiment(path)

    def test_infinite_psnr_remains_a_recorded_measurement(self):
        observation, psf, truth = research_example("cells")
        record = restore(observation, psf, RunConfig(1))
        record.ground_truth = truth
        record.output = truth.pixels
        record.quality = {"mse": 0.0, "psnr": float("inf"), "ssim": 1.0}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = export_bundle(Path(directory) / "infinite", record)
            self.assertEqual(load_experiment(path).quality, record.quality)

    def test_missing_located_tampered_inputs_and_invalid_config(self):
        observation, psf, _ = research_example("bars")
        record = restore(observation, psf, RunConfig(5))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = Path(directory) / "experiment"
            path = export_bundle(folder, record)
            moved = Path(directory) / "moved.npy"
            (folder / "observation.npy").rename(moved)
            with self.assertRaises(MissingInput):
                load_bundle(path)
            images, _, _ = load_bundle(path, {"observation": moved})
            np.testing.assert_array_equal(
                images["observation"].pixels, observation.pixels
            )
            with moved.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_bundle(path, {"observation": moved})
            data = json.loads(path.read_text())
            path.write_text(json.dumps({**data, "inputs": []}))
            with self.assertRaisesRegex(TypeError, "Invalid experiment input metadata"):
                load_bundle(path)
            data["config"]["data_range"] = 0
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_bundle(path)
            data["schema_version"] = 999
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_bundle(path)

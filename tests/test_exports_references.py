"""Reproducibility and six saved-reference regression checks."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_deconvolution.experiments.export import (
    MissingInput,
    export_bundle,
    load_bundle,
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
            config = result["record"].config
            self.assertEqual(config.iterations, 5)
            self.assertEqual(config.regularization, 0.1)
            self.assertEqual(config.epsilon, 1e-6)
            self.assertEqual(config.data_range, 1)
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

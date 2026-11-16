"""Versioned, integrity-checked experiment bundles with exact input arrays."""

from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from image_deconvolution.experiments.runs import RunConfig
from image_deconvolution.io.images import ImageData, array_hash, save_tiff


CONVENTIONS = {"boundaries": "circular", "psf_center": "floor(shape/2)",
               "psf_normalization": "unit sum", "rl_initialization": "observation copy",
               "metrics": "full-image; fixed L; Gaussian SSIM sigma=1.5, valid 11x11 population window",
               "timing": "OTF and algorithm only; excludes I/O, display, metrics and checkpoint copies"}


class MissingInput(ValueError):
    def __init__(self, role, path):
        self.role, self.path = role, path
        super().__init__(f"Missing {role} array: {path}. Locate its matching .npy file.")


def environment_versions():
    versions = {"Python": platform.python_version(), "platform": platform.platform()}
    for name in ["image-deconvolution", "numpy", "scipy", "Pillow", "PySide6-Essentials", "shiboken6"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "metadata unavailable"
    return versions


def _portable(value):
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def export_bundle(destination, record):
    """Create a new directory; never overwrite an existing experiment."""
    target = Path(destination).resolve()
    target.mkdir(parents=False, exist_ok=False)
    created = []
    try:
        inputs = {}
        for role, image in [("observation", record.observation), ("psf", record.psf),
                            ("ground_truth", record.ground_truth)]:
            if image is None:
                inputs[role] = None
                continue
            path = target / f"{role}.npy"
            created.append(path)
            with path.open("xb") as stream:
                np.save(stream, image.pixels, allow_pickle=False)
            inputs[role] = {"file": path.name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "array_sha256": array_hash(image.pixels), "identity": image.identity()}
        output = target / f"{record.config.method}_output.tif"
        created.append(output)
        save_tiff(output, record.output)
        measurements = target / "measurements.csv"
        created.append(measurements)
        with measurements.open("x", newline="", encoding="utf-8") as stream:
            columns = ["method", "iterations", "epsilon", "threshold", "regularization", "data_range",
                       "algorithm_seconds", "mse", "psnr", "ssim", "metric_note",
                       "input_mse", "input_psnr", "input_ssim", "psnr_change_db"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            row = {k: getattr(record.config, k) for k in columns[:6]}
            row.update(algorithm_seconds=record.seconds, metric_note=record.metric_note)
            row.update(record.quality or {})
            row.update({f"input_{key}": value for key, value in (record.input_quality or {}).items()})
            if record.psnr_change is not None:
                row["psnr_change_db"] = record.psnr_change
            writer.writerow(row)
        curve = target / "rl_curve.csv"
        if record.curve:
            created.append(curve)
            with curve.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["iteration", "mse", "psnr", "ssim"])
                writer.writeheader()
                writer.writerows(record.curve)
        configuration = {"schema_version": 1, "config": asdict(record.config), "inputs": inputs,
                         "conventions": CONVENTIONS, "degradation": record.degradation,
                         "environment": environment_versions(), "measurements": record.quality,
                         "input_measurements": record.input_quality,
                         "output": {"file": output.name, "dtype": "float32", "rounding": "float64 to float32 only; no clipping",
                                    "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                         "required_files": "Keep the three supplied .npy inputs (ground truth optional) with this JSON. Source arrays are exact float64 snapshots; TIFF output is rounded float32."}
        config_file = target / "experiment.json"
        created.append(config_file)
        config_file.write_text(json.dumps(_portable(configuration), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return config_file
    except Exception:
        for path in reversed(created):
            if path.is_file():
                path.unlink()
        try:
            target.rmdir()
        except OSError:
            pass
        raise


def load_bundle(configuration_path, overrides=None):
    """Validate config and exact input identities, optionally locating moved arrays."""
    path = Path(configuration_path)
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON numbers are invalid.")))
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1 or data.get("conventions") != CONVENTIONS:
        raise ValueError("Unsupported experiment version or scientific conventions.")
    if not isinstance(data.get("inputs"), dict):
        raise ValueError("Invalid experiment input metadata.")
    try:
        config = RunConfig(**data["config"])
        config.validate()
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid experiment parameters.") from exc
    images = {}
    overrides = overrides or {}
    for role in ["observation", "psf", "ground_truth"]:
        entry = data.get("inputs", {}).get(role)
        if entry is None:
            if role != "ground_truth":
                raise ValueError(f"The experiment is missing its {role} identity.")
            images[role] = None
            continue
        try:
            relative = Path(entry["file"])
            if relative.name != entry["file"] or relative.suffix != ".npy":
                raise ValueError("Experiment input filenames must be local .npy basenames.")
            input_path = Path(overrides[role]) if role in overrides else path.parent / relative
            if not input_path.is_file():
                raise MissingInput(role, input_path)
            if input_path.stat().st_size > 16*1024*1024*8 + 4096:
                raise ValueError("Experiment array exceeds the supported size.")
            if hashlib.sha256(input_path.read_bytes()).hexdigest() != entry["file_sha256"]:
                raise ValueError(f"The {role} array failed its file hash check.")
            array = np.load(input_path, allow_pickle=False)
            identity = entry["identity"]
            image = ImageData(array, identity["source"], identity["policy"], identity.get("file_sha256"))
            if array_hash(image.pixels) != entry["array_sha256"]:
                raise ValueError(f"The {role} array failed its numerical hash check.")
            images[role] = image
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid {role} metadata.") from exc
    if images["psf"].pixels.shape != images["observation"].pixels.shape:
        raise ValueError("Exported PSF dimensions must match the observation.")
    return images, config, data.get("degradation")

"""Versioned, integrity-checked experiment bundles with exact input arrays."""

import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from image_deconvolution.experiments.inspection import RL_PHASES
from image_deconvolution.experiments.runs import RunConfig, RunRecord
from image_deconvolution.io.images import ImageData, array_hash, read_image, save_tiff

CONVENTIONS = {
    "boundaries": "circular",
    "psf_center": "floor(shape/2)",
    "psf_normalization": "unit sum",
    "rl_initialization": "observation copy",
    "metrics": "full-image; fixed L; Gaussian SSIM sigma=1.5, valid 11x11 population window",
    "timing": "OTF and algorithm only; excludes I/O, display, metrics and checkpoint copies",
}


class MissingInput(ValueError):
    def __init__(self, role, path):
        self.role, self.path = role, path
        super().__init__(
            f"Missing {role} array: {path}. Locate its matching .npy file."
        )


def environment_versions():
    versions = {"Python": platform.python_version(), "platform": platform.platform()}
    for name in [
        "image-deconvolution",
        "numpy",
        "scipy",
        "Pillow",
        "PySide6-Essentials",
        "shiboken6",
    ]:
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

        def save_array(name, array):
            file = target / name
            created.append(file)
            with file.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            return {
                "file": name,
                "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "array_sha256": array_hash(array),
            }

        inputs = {}
        for role, image in [
            ("observation", record.observation),
            ("psf", record.psf),
            ("ground_truth", record.ground_truth),
        ]:
            if image is None:
                inputs[role] = None
                continue
            inputs[role] = {
                **save_array(f"{role}.npy", image.pixels),
                "identity": image.identity(),
            }
        numerical_output = save_array("output.npy", record.output)
        checkpoints = []
        for iteration, step in record.steps.items():
            checkpoints.append(
                {
                    "iteration": iteration,
                    "phases": {
                        phase: save_array(f"rl_{iteration}_{index}.npy", step[phase])
                        for index, phase in enumerate(RL_PHASES)
                    },
                }
            )
        output = target / f"{record.config.method}_output.tif"
        created.append(output)
        save_tiff(output, record.output)
        measurements = target / "measurements.csv"
        created.append(measurements)
        with measurements.open("x", newline="", encoding="utf-8") as stream:
            columns = [
                "method",
                "iterations",
                "epsilon",
                "threshold",
                "regularization",
                "data_range",
                "algorithm_seconds",
                "mse",
                "psnr",
                "ssim",
                "metric_note",
                "input_mse",
                "input_psnr",
                "input_ssim",
                "psnr_change_db",
            ]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            row = {k: getattr(record.config, k) for k in columns[:6]}
            row.update(algorithm_seconds=record.seconds, metric_note=record.metric_note)
            row.update(record.quality or {})
            row.update(
                {
                    f"input_{key}": value
                    for key, value in (record.input_quality or {}).items()
                }
            )
            if record.psnr_change is not None:
                row["psnr_change_db"] = record.psnr_change
            writer.writerow(row)
        curve = target / "rl_curve.csv"
        if record.curve:
            created.append(curve)
            with curve.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["iteration", "mse", "psnr", "ssim"]
                )
                writer.writeheader()
                writer.writerows(record.curve)
        configuration = {
            "schema_version": 2,
            "config": asdict(record.config),
            "inputs": inputs,
            "conventions": CONVENTIONS,
            "degradation": record.degradation,
            "environment": environment_versions(),
            "measurements": record.quality,
            "input_measurements": record.input_quality,
            "result": {
                "array": numerical_output,
                "algorithm_seconds": record.seconds,
                "metric_note": record.metric_note,
                "inspection_note": record.inspection_note,
                "curve": record.curve,
                "checkpoints": checkpoints,
            },
            "output": {
                "file": output.name,
                "dtype": "float32",
                "rounding": "float64 to float32 only; no clipping",
                "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            "required_files": "Keep this whole folder. NPY files preserve exact float64 inputs, output and captured RL phases; TIFF output is rounded float32. Reopen restores the saved run without executing restoration.",
        }
        config_file = target / "experiment.json"
        created.append(config_file)
        config_file.write_text(
            json.dumps(_portable(configuration), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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


def _manifest(configuration_path):
    path = Path(configuration_path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Experiment JSON exceeds the supported size.")
    data = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON numbers are invalid.")
        ),
    )
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") not in (1, 2)
        or data.get("conventions") != CONVENTIONS
    ):
        raise ValueError("Unsupported experiment version or scientific conventions.")
    return data


def load_bundle(configuration_path, overrides=None):
    """Validate config and exact input identities, optionally locating moved arrays."""
    path = Path(configuration_path)
    data = _manifest(path)
    if not isinstance(data.get("inputs"), dict):
        raise TypeError("Invalid experiment input metadata.")
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
                raise ValueError(
                    "Experiment input filenames must be local .npy basenames."
                )
            input_path = (
                Path(overrides[role]) if role in overrides else path.parent / relative
            )
            if not input_path.is_file():
                raise MissingInput(role, input_path)
            if input_path.stat().st_size > 16 * 1024 * 1024 * 8 + 4096:
                raise ValueError("Experiment array exceeds the supported size.")
            if (
                hashlib.sha256(input_path.read_bytes()).hexdigest()
                != entry["file_sha256"]
            ):
                raise ValueError(f"The {role} array failed its file hash check.")
            array = np.load(input_path, allow_pickle=False, mmap_mode="r")
            identity = entry["identity"]
            if (
                array.dtype != np.float64
                or array.ndim != 2
                or list(array.shape) != identity.get("shape")
            ):
                raise ValueError(f"Invalid saved {role} array shape or type.")
            image = ImageData(
                array,
                identity["source"],
                identity["policy"],
                identity.get("file_sha256"),
            )
            if array_hash(image.pixels) != entry["array_sha256"]:
                raise ValueError(f"The {role} array failed its numerical hash check.")
            images[role] = image
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid {role} metadata.") from exc
    if images["psf"].pixels.shape != images["observation"].pixels.shape:
        raise ValueError("Exported PSF dimensions must match the observation.")
    psf = images["psf"].pixels
    if np.any(psf < 0) or not np.isclose(psf.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Exported PSF must be nonnegative and normalized to unit sum.")
    return images, config, data.get("degradation")


def _checked_file(root, entry, suffix, limit):
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        raise TypeError("Invalid saved result file metadata.")
    name = entry["file"]
    if Path(name).name != name or Path(name).suffix != suffix:
        raise ValueError("Saved result filenames must be local basenames.")
    file = root / name
    if not file.is_file():
        raise ValueError(
            f"Missing saved result file: {name}. Restore the complete export folder."
        )
    if file.stat().st_size > limit:
        raise ValueError("Saved result file exceeds the supported size.")
    if hashlib.sha256(file.read_bytes()).hexdigest() != entry.get("file_sha256"):
        raise ValueError(f"The saved {name} failed its file hash check.")
    return file


def _saved_array(root, entry, shape):
    file = _checked_file(root, entry, ".npy", 4096 * 4096 * 8 + 4096)
    mapped = np.load(file, allow_pickle=False, mmap_mode="r")
    if mapped.shape != shape or mapped.dtype != np.float64:
        raise ValueError(
            "Saved result arrays must match the input shape and float64 type."
        )
    if not np.isfinite(mapped).all() or array_hash(mapped) != entry.get("array_sha256"):
        raise ValueError(
            "Saved result array failed its numerical hash or finiteness check."
        )
    array = np.array(mapped)
    array.setflags(write=False)
    return array


def _number(value, name, allow_infinity=False):
    if allow_infinity and value == "Infinity":
        return float("inf")
    if type(value) not in (float, int) or not np.isfinite(value):
        raise ValueError(f"Invalid saved {name}.")
    return float(value)


def _quality(values):
    if values is None:
        return None
    if not isinstance(values, dict) or set(values) != {"mse", "psnr", "ssim"}:
        raise ValueError("Invalid saved quality measurements.")
    result = {
        key: _number(values[key], key, allow_infinity=key == "psnr")
        for key in ("mse", "psnr", "ssim")
    }
    if result["mse"] < 0 or not -1 <= result["ssim"] <= 1:
        raise ValueError("Invalid saved quality measurement range.")
    return result


def _curve(rows, config):
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("Saved RL curves support at most 32 measurements.")
    if rows and (config.method != "RL" or not config.inspect):
        raise ValueError("Saved quality curves require a captured RL run.")
    result, previous = [], 0
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Invalid saved RL curve row.")
        iteration = row.get("iteration")
        if type(iteration) is not int or not previous < iteration <= config.iterations:
            raise ValueError("Invalid saved RL curve iteration.")
        values = _quality({key: row.get(key) for key in ("mse", "psnr", "ssim")})
        result.append({"iteration": iteration, **values})
        previous = iteration
    return result


def load_experiment(configuration_path, overrides=None):
    """Restore a validated saved run without executing an algorithm or metrics."""
    path = Path(configuration_path)
    data = _manifest(path)
    images, config, degradation = load_bundle(path, overrides)
    shape = images["observation"].pixels.shape
    output_file = _checked_file(
        path.parent, data.get("output"), ".tif", 4096 * 4096 * 8 + 4096
    )
    steps, curve, inspection_note = {}, [], ""
    if data["schema_version"] == 2:
        result = data.get("result")
        if not isinstance(result, dict):
            raise ValueError("Missing saved restoration result metadata.")
        output = _saved_array(path.parent, result.get("array"), shape)
        seconds = _number(result.get("algorithm_seconds"), "algorithm time")
        note = result.get("metric_note")
        inspection_note = result.get("inspection_note", "")
        curve = _curve(result.get("curve"), config)
        checkpoints = result.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) > 8:
            raise ValueError("Saved RL inspection supports at most eight checkpoints.")
        if len(checkpoints) * 5 * output.nbytes > 64 * 1024 * 1024:
            raise ValueError("Saved RL checkpoints exceed 64 MiB.")
        if checkpoints and (config.method != "RL" or not config.inspect):
            raise ValueError("Saved checkpoints require a captured RL run.")
        previous = 0
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                raise TypeError("Invalid saved RL checkpoint.")
            iteration = checkpoint.get("iteration")
            phases = checkpoint.get("phases")
            if (
                type(iteration) is not int
                or not previous < iteration <= config.iterations
            ):
                raise ValueError("Invalid saved RL checkpoint iteration.")
            if not isinstance(phases, dict) or set(phases) != set(RL_PHASES):
                raise ValueError("A saved RL checkpoint must include all five phases.")
            steps[iteration] = {
                phase: _saved_array(path.parent, phases[phase], shape)
                for phase in RL_PHASES
            }
            previous = iteration
    else:
        # Version 1 only stored a rounded TIFF, plus timing and curves in CSVs.
        output = read_image(output_file).pixels
        if output.shape != shape:
            raise ValueError("Saved TIFF dimensions must match the observation.")
        measurements = path.parent / "measurements.csv"
        if measurements.stat().st_size > 64 * 1024:
            raise ValueError("Saved measurement CSV exceeds the supported size.")
        with measurements.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0].get("method") != config.method:
            raise ValueError("Invalid legacy measurement CSV.")
        try:
            seconds = _number(float(rows[0]["algorithm_seconds"]), "algorithm time")
            for key in (
                "iterations",
                "epsilon",
                "threshold",
                "regularization",
                "data_range",
            ):
                if float(rows[0][key]) != getattr(config, key):
                    raise ValueError("Legacy CSV parameters do not match the JSON.")
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid legacy measurement CSV.") from exc
        note = rows[0].get("metric_note", "")
        curve_file = path.parent / "rl_curve.csv"
        if curve_file.is_file():
            if curve_file.stat().st_size > 64 * 1024:
                raise ValueError("Saved RL curve CSV exceeds the supported size.")
            try:
                with curve_file.open(newline="", encoding="utf-8") as stream:
                    curve = _curve(
                        [
                            {
                                "iteration": int(row["iteration"]),
                                **{
                                    key: "Infinity"
                                    if row[key] == "inf"
                                    else float(row[key])
                                    for key in ("mse", "psnr", "ssim")
                                },
                            }
                            for row in csv.DictReader(stream)
                        ],
                        config,
                    )
            except (KeyError, TypeError) as exc:
                raise ValueError("Invalid legacy RL curve CSV.") from exc
        if config.method == "RL":
            inspection_note = "This legacy export has no RL checkpoints. Enable Capture RL steps and rerun to inspect intermediate images."
        note += (
            " Reopened legacy float32 TIFF; saved measurements precede TIFF rounding."
        )
    if seconds < 0 or not isinstance(note, str) or not isinstance(inspection_note, str):
        raise ValueError("Invalid saved result timing or notes.")
    values = _quality(data.get("measurements"))
    input_values = _quality(data.get("input_measurements"))
    truth = images["ground_truth"]
    if (values is not None or input_values is not None or curve) and (
        truth is None
        or truth.pixels.shape != shape
        or min(shape) <= 10
        or truth.pixels.min() < 0
        or truth.pixels.max() > config.data_range
    ):
        raise ValueError("Saved quality measurements require suitable ground truth.")
    if not steps and config.method == "RL" and not inspection_note:
        inspection_note = "This saved run has no RL checkpoints. Enable Capture RL steps and rerun to inspect intermediate images."
    return RunRecord(
        observation=images["observation"],
        psf=images["psf"],
        config=config,
        output=output,
        seconds=seconds,
        ground_truth=truth,
        quality=values,
        metric_note=note,
        steps=steps,
        curve=curve,
        degradation=degradation,
        input_quality=input_values,
        saved_from=str(path.resolve()),
        inspection_note=inspection_note,
    )

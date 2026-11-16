"""New Python calculations against saved DL2 TIFFs; never executes ImageJ."""

import csv
import hashlib
import json

import numpy as np

from image_deconvolution.experiments.inspection import quality
from image_deconvolution.experiments.runs import RunConfig, restore
from image_deconvolution.io.images import read_image, sample_path


def verify_resources():
    manifest = json.loads(sample_path("manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        digest = hashlib.sha256(sample_path(entry["file"]).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(
                f"Bundled research resource failed its hash check: {entry['file']}"
            )


def research_example(name):
    if name not in {"cells", "bars", "fibers"}:
        raise ValueError("Choose cells, bars, or fibers.")
    return (
        read_image(sample_path(f"{name}_s2_p200_r0_input.tif")),
        read_image(sample_path("psf_s2.tif")),
        read_image(sample_path(f"{name}_gt.tif")),
    )


def compare_references():
    verify_resources()
    with sample_path("imagej_metrics.csv").open("r", encoding="utf-8-sig") as stream:
        saved = {(r["image"], r["method"]): r for r in csv.DictReader(stream)}
    rows = []
    for name in ["cells", "bars", "fibers"]:
        observation, psf, truth = research_example(name)
        for method in ["RL", "TRIF"]:
            record = restore(observation, psf, RunConfig(5, method=method), truth)
            reference = read_image(sample_path(f"dl2_{name}_{method}.tif"))
            difference = record.output - reference.pixels
            reference_quality, _ = quality(truth, reference.pixels, 1)
            rows.append(
                {
                    "image": name,
                    "method": method,
                    "record": record,
                    "reference": reference,
                    "difference": difference,
                    "max_abs": float(np.max(np.abs(difference))),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "reference_quality_python": reference_quality,
                    "saved_imagej_mse": float(saved[(name, method)]["mse_imagej"]),
                    "saved_imagej_psnr": float(saved[(name, method)]["psnr_imagej"]),
                }
            )
    return rows

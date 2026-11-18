"""Read grayscale pixels explicitly; never measure display conversions."""

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def array_hash(array):
    a = np.asarray(array, dtype="<f8", order="C")
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class ImageData:
    pixels: np.ndarray
    source: str
    policy: str = "preserve"
    file_sha256: str | None = None
    nominal_range: float | None = None

    def __post_init__(self):
        pixels = validate_image(self.pixels).copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "pixels", pixels)

    def identity(self):
        return {
            "source": self.source,
            "policy": self.policy,
            "file_sha256": self.file_sha256,
            "array_sha256": array_hash(self.pixels),
            "shape": list(self.pixels.shape),
        }


def validate_image(array):
    a = np.asarray(array, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("Select a two-dimensional grayscale image, at least 2 x 2 pixels.")
    if a.size > 4096 * 4096:
        raise ValueError("Images are limited to 16 megapixels in this version.")
    if not np.isfinite(a).all():
        raise ValueError("The image contains NaN or infinite values.")
    return a


def read_image(path, policy="preserve"):
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff", ".png"}:
        raise ValueError("Select a grayscale TIFF or PNG file.")
    if policy not in {"preserve", "normalize_integer"}:
        raise ValueError("Unknown intensity conversion policy.")
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Stacks are unsupported. Select a single 2D grayscale image.")
            if image.mode not in {"L", "I", "F", "I;16", "I;16L", "I;16B"}:
                raise ValueError(
                    "Only grayscale input is supported. Convert color or palette images explicitly before loading."
                )
            if image.width * image.height > 4096 * 4096:
                raise ValueError("Images are limited to 16 megapixels in this version.")
            raw = np.array(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image: {path.name}. {exc}") from exc
    pixels = validate_image(raw)
    if policy == "normalize_integer":
        if raw.dtype.kind != "u":
            raise ValueError(
                "Integer normalization requires an unsigned 8-bit or 16-bit image; use Preserve values for float or signed data."
            )
        pixels = pixels / np.iinfo(raw.dtype).max
    nominal_range = (
        (1.0 if policy == "normalize_integer" else float(np.iinfo(raw.dtype).max)) if raw.dtype.kind == "u" else None
    )
    return ImageData(
        pixels,
        str(path.resolve()),
        policy,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        nominal_range,
    )


def prepare_psf(array, shape):
    """Center-pad a PSF to image shape and normalize its finite positive mass."""
    h = validate_image(array)
    if any(a > b for a, b in zip(h.shape, shape)):
        raise ValueError("The PSF dimensions must not exceed the image dimensions.")
    total = float(h.sum())
    if np.any(h < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("The PSF must be nonnegative and have a finite positive sum.")
    padded = np.zeros(shape, dtype=np.float64)
    row, col = (b // 2 - a // 2 for a, b in zip(h.shape, shape))
    padded[row : row + h.shape[0], col : col + h.shape[1]] = h / total
    return padded


def save_tiff(path, array, overwrite=False):
    """Write float32 TIFF without clipping; fail instead of overflowing."""
    a = validate_image(array)
    if np.any(np.abs(a) > np.finfo(np.float32).max):
        raise ValueError("Values exceed the float32 TIFF range.")
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Use a .tif or .tiff destination.")
    with path.open("wb" if overwrite else "xb") as stream:
        Image.fromarray(a.astype(np.float32)).save(stream, format="TIFF")


def sample_path(name):
    return files("image_deconvolution").joinpath("resources", "samples", name)

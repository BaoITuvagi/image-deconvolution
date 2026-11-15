# Image Deconvolution

I am building an offline Windows desktop app to study thresholded inverse filtering,
zeroth-order Tikhonov regularization, and Richardson-Lucy deconvolution. The app
supports the group 07 image processing course project. I develop the desktop app
independently and adapt the group's existing numerical research with
[source attribution](docs/attribution.md).

The first restoration workflow is available: open a grayscale degraded TIFF/PNG,
generate or import a centered PSF, run Richardson-Lucy in a background worker,
inspect synchronized views, and save numerical float32 TIFF. Load saved cells
example is a quick starting point. The [plan](docs/plan.md) tracks later stages.

## Development setup

Use 64-bit Python 3.13 on Windows. Setup was verified with Python 3.13.6; other
Python minor versions are currently excluded by the package metadata. From the
repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -B -m image_deconvolution
```

The window closes through **File > Exit**, **Ctrl+Q**, or the title-bar
close button. After installation, `.venv\Scripts\image-deconvolution.exe` is an
alternative launcher. Activating the environment is optional. Installation
downloads dependencies; running the installed app requires no server or network.
Do not install these dependencies into the global Python environment.

Direct runtime dependencies are pinned in `pyproject.toml`: NumPy 2.3.3,
SciPy 1.16.2, Pillow 12.3.0, and PySide6 6.10.0. PySide6 pins its matching Qt
components and Shiboken. The build backend is setuptools 82.0.1. These are the
verified app environment versions, not a claim to reproduce the original research
environment or its performance measurements.

## Numerical validation

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.venv\Scripts\python.exe -m pip check
```

The maintained checks preserve the original convolution, adjoint, SSIM identity,
RL identity, and Tikhonov analytic checks. They also cover IF threshold behavior,
odd rectangular PSF centering, fixed-range PSNR, RL checkpoint independence, and
importing the numerical core without Qt. They use Python's standard `unittest`.

The numerical API currently assumes valid 2D floating-point inputs with matching
shapes. Use float64 for processing, a centered normalized full-image PSF, circular
boundaries, and normalized intensities with PSNR range L = 1. RL starts from a
copy of the observed image and uses epsilon = 1e-6 by default. Tikhonov uses L = I.
The core does not clip or rescale restoration output. Workflow validation covers
finite grayscale inputs, PSF mass and size, nonnegative observations, and RL
parameters. Color images and stacks are rejected. Choose Preserve values or
explicit unsigned-integer normalization before loading an image. PSFs are
center-padded and normalized to unit sum. Display changes do not affect exports.

## Source layout and samples

- `src/image_deconvolution/core/`: numerical routines without Qt dependencies.
- `src/image_deconvolution/ui/`: the minimal desktop window.
- `src/image_deconvolution/resources/samples/`: the saved synthetic cells input,
  matching sigma-2 PSF, and provenance manifest for the next workflow.
- `tests/`: numerical regression checks.
- `docs/`: plan and research attribution.

The samples are package data. They can be located independently of the checkout
or current directory through `importlib.resources.files("image_deconvolution")`
and its `resources/samples` child. Load saved cells example opens both files.
No external research folder is needed to install, test, or launch the app.

## Research scope and limitations

The baseline concerns synthetic 2D grayscale images, known PSFs, and circular
boundaries. The original 81 cases combine three source images, three blur widths,
three photon levels, and three seeds; they are not 81 independent scenes. Metrics
require compatible ground truth and explicit intensity conventions.

Fiji/ImageJ with DeconvolutionLab2 remains the external reference platform.
Stage 1 does not run ImageJ, verify new plugin comparisons, or provide a packaged
Windows distribution. I do not claim a new restoration algorithm or superiority
over DeconvolutionLab2. No project-wide license has been assigned to the reused
research; attribution does not grant redistribution rights. External JARs, source
reports, full research datasets, and build outputs are not bundled.

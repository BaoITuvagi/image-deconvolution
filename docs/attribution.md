# Research provenance and numerical conventions

I develop this desktop application independently. I adapted the numerical code
and baseline tests from the existing group 07 course research package named
`Deconvolution_XLA`, supplied through another group member's copy. This reuse does
not imply that other group members developed the desktop app, or that I solely
authored the group's research. Individual source-code authorship is not established
by that copy, so no individual author or license is inferred here.

The following source-relative paths identify the imported material. SHA-256 values
record the inspected source bytes; they are provenance records, not runtime paths.

| Original file | SHA-256 | Use |
| --- | --- | --- |
| `code/algorithms.py` | `82345d36b2be43a0584a9c7437dda16816167affdd93dc5405cb84fffc51920f` | Adapted into `core/algorithms.py`; unused phantom generation and an unused signal import omitted |
| `code/test_algorithms.py` | `407d0f34db62426f1bf476df76841ef136ba24a1f2f46313e1c2e97f74f44377` | Five original numerical checks retained in `tests/test_algorithms.py`, with additional regression checks |
| `data/cells_s2_p200_r0_input.tif` | `28c51869c95a77caaee9de85b04ad7db2ca513ce452aee5670dbd5d3655959d5` | Unchanged saved input for the next RL workflow |
| `data/psf_s2.tif` | `dd2a2f9f5e917d2d11bf0bf02542039c2d0c6f4cfa35611347d288fc76fceb53` | Unchanged matching PSF |

The initial two TIFFs are included as small package resources. The input is a
256 x 256 synthetic cells image blurred with a Gaussian sigma of 2 pixels and
Poisson noise at photon level 200, seed 0. TIFF storage is float32. The input was
copied, not regenerated; the same seed on another platform need not reproduce
the saved pixels. The manifest beside the TIFFs records their identities. Ground
truth, restored outputs, and plugin references are deferred until needed.

Stages 3-4 also import unchanged `data/cells_gt.tif`, `data/bars_gt.tif`, and
`data/bars_s2_p200_r0_input.tif` for quality evaluation and the late-iteration
bars demonstration. Their original paths and SHA-256 hashes are recorded in the
package sample manifest. They share the same synthetic research provenance;
none of these images is presented as a real-world photograph.

## Preserved calculations

- A full-image centered, normalized PSF becomes an OTF using `ifftshift` and
  `rfft2`. Convolution and its adjoint use circular boundaries.
- Thresholded IF divides only frequencies with `abs(H) >= tau`.
- Tikhonov uses `conj(H) * Y / (abs(H)**2 + lambda)` with identity regularization
  L = I. It is not the Laplacian variant.
- RL initializes from a copy of the observation, divides by a prediction floored
  at epsilon = 1e-6, and applies the conjugate-OTF backprojection. Checkpoints are
  independent copies. The optional redundant-transform baseline is retained only
  for compatibility with the research, not as a performance claim.
- MSE and PSNR operate on the numerical arrays with fixed range L = 1. SSIM uses
  an 11 x 11 Gaussian window, sigma 1.5, population covariance, and the valid
  interior after excluding five pixels on each edge. Images must have matching
  shapes and both dimensions greater than 10.
- No display conversion, clipping, or output renormalization is added to the core.
  The Stage 1 API assumes valid floating-point inputs; user-input validation is
  part of the next workflow.

The algorithms were reformatted and documented without changing their baseline
equations, defaults, or return conventions. The original five checks were rerun
before adaptation. Further validation is described in the Stage 1 PR and plan.

## External tools and rights

NumPy, SciPy, Pillow, and PySide6 are installed as third-party dependencies and
retain their respective terms. Fiji/ImageJ and DeconvolutionLab2 are reference
tools; no JAR or plugin source is included. This application does not replace
their verification role. No new ImageJ comparison was executed for Stage 1.

No project-wide license is assigned by this adaptation. Source attribution does
not establish permission to relicense the research or redistribute external
binaries; rights must be confirmed before a distribution release.

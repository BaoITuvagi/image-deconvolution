# User guide

## Start the Windows app

Keep the complete `Image Deconvolution` folder together and open
`ImageDeconvolution.exe`. The `_internal` folder contains Python, native libraries,
and saved examples. No development Python, Fiji, server, account, or network is
required to run the desktop workflow. This is an unsigned local evaluation build.
Do not publish or redistribute the research assets before confirming their rights.

The sidebar has four expandable sections: Inputs & simulation, Restoration,
Display & history, and Experiment files. PSF and intensity fields appear only
inside Inputs or Restoration, without a separate frame. Inputs places them after
Noise seed and before Generate; Restoration places them above the algorithm
settings. The row order is PSF mode, Open PSF, Gaussian sigma, then Intensity
range L. Open PSF is a clickable filename field: it is blank before a file is
selected and shows the filename afterward. Click it or focus it and press Enter
or Space to open the file picker. It is dimmed and disabled in Gaussian mode;
Gaussian sigma is dimmed and disabled in Imported PSF mode. Both panels use the
same settings; switching panels retains their values. Display & history and
Experiment files hide these fields. The main tabs are Compare & history,
RL inspector, and Saved references. Long sidebar sections can be scrolled.
Field labels end with a colon and stay beside their controls. The sidebar
reserves enough width for inline fields, including at larger text sizes.
Loading policy and Saved example come first, followed by the image-opening and
ground-truth buttons, photon level, seed, PSF/intensity fields, and Generate.
Image information is grouped at the bottom: file, dimensions, numerical pixel
range, image scale when known, and ground-truth status. PSF summary text and
standalone instructions below Generate or in Image information are omitted.
A display range labeled "on load" records the initial viewing suggestion;
current display settings remain in Display & history.

## Restore a saved example

1. In Inputs & simulation, choose Cells — reference and click Load beside
   the selector. The sidebar stays on Inputs & simulation. This opens the saved
   degraded float TIFF, its matching sigma-2 PSF, and ground truth. It does not
   regenerate noise or run restoration. Cells, Bars, and Fibers each offer two
   presets for the saved sigma-2, photon-200, seed-0 TIFFs. The reference presets
   load IF threshold 0.1, TRIF lambda 0.1, RL 5 iterations, epsilon 1e-6, and L=1
   for reference comparisons. The Report — P=200 presets load the quality
   parameters selected at P=200: IF 0.3, TRIF 0.1, RL 2, epsilon 1e-6, and L=1.
   To demonstrate report Table 5.5, choose Cells — Report — P=200 and click Load,
   then run the three-method comparison in Restoration.
2. Expand Restoration. Choose RL, IF, or TRIF, and edit the method's parameters.
   The PSF and L fields retain the settings from Inputs. Gaussian sigma is
   used only when PSF is Gaussian. Imported centered PSFs are center-padded to
   image size and normalized to unit sum.
   Inactive parameters are dimmed. Small decimal parameters use adaptive arrow
   steps. To configure a three-method comparison, select each method and edit its
   retained settings before clicking Compare.
3. Click Run selected method or Compare IF, TRIF, and RL. Processing runs in a
   worker; the controls stay disabled until it finishes. The views remain usable.
4. Select a history row with the mouse or arrow keys to inspect that run's
   original observation, ground truth, output, measurements, and configuration.
   Two Additional results selectors above the images list only runs with identical
   observed pixels. After a three-method comparison, all three outputs appear
   immediately. The five views use one display range. Each output identifies its
   method and parameters; the history table also identifies the input image.
   Loading a new image or editing ground truth shows the current inputs immediately
   and retains history. View current inputs returns to the next run's inputs at
   any time. History selection does not replace the controls for the next run;
   an earlier-configuration notice remains visible when appropriate.
   Degraded-input and selected-output MSE/PSNR/SSIM use the same ground truth and
   L. PSNR change is output minus input; a negative value means lower PSNR. These
   exploratory results do not select benchmark parameters. Undefined infinity
   differences show N/A.
5. Click Save selected output TIFF to save float32 pixels, including negative
   values or values above 1. This rounds float64 to float32; it never saves the
   display screenshot. An existing file requires the native save dialog's
   overwrite confirmation.

History is limited to eight runs or approximately 256 MiB of counted input/output/
checkpoint arrays, except a single larger run. Oldest entries are dropped first.
History is in memory; export a needed run before closing or clearing it.
Clear run history asks for confirmation and keeps saved files.

## Open your own images and simulate degradation

Open degraded image accepts a single 2D grayscale TIFF or PNG, with finite values,
at least 2 x 2 pixels and no more than 16 megapixels. Color, palette images, and
stacks are rejected. Convert them explicitly in an external tool first.

Choose the loading policy before opening an image: Preserve values retains raw
numerical intensities; Normalize unsigned integers to [0, 1] divides 8-/16-bit
unsigned values by their dtype maximum. Float and signed inputs require Preserve
values. The input tooltip reports the applied policy. On load, the shared display
starts at the unsigned-integer scale (255 or 65535), or [0, 1] for normalized
integers; other custom inputs use input min/max. Saved examples use [0, 1].
This suggestion changes viewing only: numerical arrays and intensity L are kept.
For preserved integers, explicitly set L to the image's declared scale, or reopen
with normalization. Display & history can change the shared display range.

Open optional ground truth uses the same loading policy. In Inputs or Restoration,
set Intensity range L to the ground truth's declared intensity scale (for example,
1 for normalized research pixels or 255 for preserved 8-bit pixels). Metrics are
N/A with an explanation if truth is missing, dimensions differ, either dimension
does not exceed 10, or truth lies outside [0, L]. Clear ground truth removes it
for subsequent runs; historical measurements retain their original truth.

For simulation, Open source for simulation loads an undegraded source into
Current ground truth only. Current observation stays empty with a Generate prompt;
Run selected method and Compare remain disabled. Opening a new source clears the
current observation and restoration preview while retaining history. Historical
runs can still be inspected; View current inputs returns to the pending source.
Configure photon level, seed, PSF and L in Inputs & simulation, then explicitly
click Generate / regenerate degradation. On success, Current observation shows the degraded image,
the source stays as ground truth, and restoration becomes available. If generation
fails, no observation is created and the source remains available for correction.
Saved examples and Open degraded image provide observations immediately.
The model is circular blur followed by Poisson sampling of
`max(blur(source)/L, 0) * photons`, scaled back by `L/photons`. The maximum only
removes tiny negative FFT roundoff at theoretical zeros. The source must lie in
[0, L]. L supplies the intensity scale for Poisson noise and quality metrics; the
blur convolution itself uses the PSF. Changing PSF, L, method parameters, photon
level, or seed alone never regenerates the observation. Restoration uses the
currently selected PSF, so a different PSF can be tested against the same fixed
observation. Historical runs retain their original PSF and L.
Saved TIFFs are preferred for report reproduction;
equal seeds across platforms do not promise equal noise pixels.

## Inspect Richardson-Lucy

Enable Capture RL steps before an RL run, then open RL inspector. Choose a
captured update and one of its five phases: current estimate, blurred prediction,
observed-to-predicted ratio, adjoint correction, or updated estimate. Update 1
starts from x(0)=the observation. The ratio uses `max(prediction, epsilon)`;
the correction is circular convolution with the adjoint PSF.

At most eight complete checkpoints are retained within 64 MiB. If one checkpoint
would exceed that budget, turn off capture to restore the large image. The curve
uses up to 32 selected measurements including the final iteration. Choose PSNR or
SSIM separately. Infinite PSNR is preserved in measurements but omitted from the
finite plot. These curves are exploratory and do not replace independently
selected benchmark parameters.

To explore iterations, load Bars — reference, then edit RL iterations in
Restoration (for example, 5, 20, 80, or 200) and run again. Its quality can rise and
later fall, illustrating why a sharper-looking late estimate need not have better
PSNR/SSIM. These exploratory runs do not replace the five-iteration reference.
Intermediate phases have their own clearly labeled display-only min/max range;
comparison views retain the shared range. Zoom and pan affect inspection, not
the full-image measurement region.
Fit all images resets the comparison, inspector, and saved-reference image views.
The phase readout explicitly labels PSNR/SSIM as quality of the updated estimate,
even when the displayed phase is the prediction, ratio, correction, or prior estimate.

## Compare saved DeconvolutionLab2 references

Saved references > Compare six saved RL / TRIF references performs new Python
calculations on the bundled original TIFF inputs and compares them to saved DL2
outputs for cells, bars, and fibers. It does not execute ImageJ, Fiji, or a JAR.
The comparison configuration stays fixed regardless of the selected preset.

The configuration is sigma 2, photon level 200, seed 0, RL 5 iterations or TRIF
lambda 0.1, PSF normalization 1, circular boundaries, epsilon 1e-6, RL initialized
from the observation, Academic FFT in the saved plugin, no padding/apodization,
and intact float output. All TIFFs and metadata are checked against recorded
source hashes. The table separates max absolute difference/RMSE agreement from
restoration quality. ImageJ MSE/PSNR columns are saved measurements; SSIM is
calculated in Python. The signed difference image has its own labeled range.
Show reference details expands the configuration/provenance and additional RMSE,
saved ImageJ MSE, and Python SSIM of saved DL2 columns. The compact table keeps
agreement and quality visible without the longer technical explanation.

The largest reproduced absolute difference in the verified environment is
8.104919255735865e-7. The 1e-6 baseline is configuration-specific. IF has not been
verified against a different inverse-filter plugin variant. No superiority over
DeconvolutionLab2 or a new restoration algorithm is claimed.

## Export and reopen an experiment

Select a history run, expand Experiment files, and choose Export selected
experiment. Choose an existing **parent folder**, then enter a **new folder name**
in the second dialog. Canceling either dialog creates nothing. Existing
destinations are refused. The new folder contains:

- `experiment.json`: schema version, method parameters, scientific conventions,
  source identities/hashes, applicable noise settings and environment versions.
- `observation.npy`, `psf.npy`, and optional `ground_truth.npy`: exact float64
  numerical input snapshots. These files are required to rerun the JSON.
- `<method>_output.tif`: numerical float32 output without clipping/rescaling.
- `measurements.csv` and, when available, `rl_curve.csv`: values from that run.
  Measurements include the degraded-input baseline and PSNR change; JSON also
  records the input measurements. CSV curve rows still describe updated RL estimates.

Reopen experiment JSON checks the version, conventions, dimensions, and file/
numerical hashes and restores the controls. If an input array has moved, locate
the matching .npy file when prompted. A changed array is rejected. Click Run
selected method explicitly to rerun. JSON does not embed pixels. In this Windows
environment, reopen/rerun differences from repeated PSF normalization are tested
with absolute tolerance 1e-12, independently of plugin comparison tolerance.

## Timing, limits, and recovery

Time values cover OTF construction and numerical restoration only. They exclude
file I/O, validation, metric computation, checkpoint copies, and rendering; they
are not full UI duration or a cross-library speed claim. All processing uses the
local CPU. These single-run timings differ from the research benchmark's warm-up
and repeated-run medians. The report's 22.9% time reduction / 1.30x speedup belongs
to its research benchmark, not a verified speedup of this desktop change.
The legacy `cache=False` code path inverse/forward-transforms H in each iteration;
it is not the report's baseline that rebuilds the OTF from the spatial PSF.
Long runs have no cancellation in this version. Closing is blocked
while a worker runs; wait for completion. If a worker fails, the app reports its
English error and re-enables controls without relabeling an older result.

The implementation currently targets 2D grayscale known-PSF restoration with
circular boundaries. It does not support blind deconvolution, color restoration,
3D stacks, alternate boundaries, automatic JAR execution, or real-image efficacy
claims. The Windows build is tested on the current machine, not a separate clean
Windows installation. See the release notes for the precise verification scope.

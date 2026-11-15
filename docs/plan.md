# Desktop deconvolution app implementation plan

Status: Stages 1-6 implemented and verified on Windows with Python 3.13.6.
Stage 7 packaging remains under the delegated implementation workflow.

## Purpose

I will build an offline desktop app to investigate image restoration with thresholded inverse filtering (IF), zeroth-order Tikhonov (TRIF), and Richardson-Lucy (RL). I will use my existing course research as the numerical foundation and Fiji/ImageJ with DeconvolutionLab2 as the reference for verification.

I am developing the app independently. I will retain accurate attribution for research, code, data, and external tools that I reuse. The app will support the group 07 image processing assignment through reproducible demonstrations, quantitative evaluation, and explanations of the algorithms.

The main demonstration will follow this sequence:

1. Load a source image and PSF, or select a saved research example.
2. Generate a controlled degraded image when ground truth is available.
3. Run and compare the three restoration methods.
4. Inspect RL iterations and their effect on image quality.
5. Compare compatible outputs against saved DeconvolutionLab2 results.
6. Export the results and the configuration needed to reproduce them.

## Confirmed technical decisions

| Area | Decision |
| --- | --- |
| Application | Local desktop app, initially targeting Windows |
| Interface | Python with PySide6 |
| Numerical processing | Python with NumPy and SciPy |
| Image I/O | Reuse Pillow where it preserves the required image values |
| Existing implementation | Reuse and adapt the research algorithms rather than replacing them with plugin calls |
| Execution | Offline processing on the local CPU; no separate server |
| Language | English throughout the app, source documentation, and GitHub work |
| Reference platform | Fiji/ImageJ and DeconvolutionLab2 |
| Initial scientific scope | Two-dimensional grayscale images, known PSFs, and circular boundaries |

Stage 1 pins NumPy 2.3.3, SciPy 1.16.2, Pillow 12.3.0, and PySide6 6.10.0,
with setuptools 82.0.1 as the build backend. The initial app environment targets
64-bit Python 3.13 and was verified with Python 3.13.6 on Windows; the original
research environment used Python 3.12. The checks recorded below establish the
Stage 1 scope only, not standalone Windows distribution compatibility.

## Scope and priorities

### Initial working milestone

The first usable workflow will be:

**Open a degraded image -> load or generate a PSF -> run RL -> inspect and save the restored image.**

I will complete this workflow before expanding the app into a larger experiment interface.

### First complete version

- Load source images, degraded images, optional ground truth, and PSFs.
- Generate Gaussian PSFs and controlled Poisson degradation with a recorded seed.
- Run IF, TRIF, and RL with explicit parameters.
- Compare images using a shared display range and synchronized regions.
- Calculate restoration metrics only when compatible ground truth is available.
- Inspect selected RL iterations and intermediate calculations.
- Plot RL quality against iteration count when ground truth exists.
- Reproduce selected saved research examples and compare with reference outputs.
- Export float TIFF results, CSV measurements, and JSON configurations.
- Provide a documented, verified Windows desktop build.

### Extensions after the first complete version

- Broader IF threshold, Tikhonov lambda, and PSF mismatch sweeps.
- Direct execution of the external DeconvolutionLab2 JAR.
- Additional boundary models, image formats, and datasets after separate validation.

Accounts, cloud storage, a web interface, an API server, blind deconvolution, color restoration, and three-dimensional stacks are outside the first version. I will not claim that results on synthetic images establish effectiveness on arbitrary real photographs.

## Proposed architecture

I will keep numerical processing independent of Qt and introduce modules only when their responsibilities are needed.

```text
src/image_deconvolution/
    __main__.py       Application entry point
    core/             PSF, Fourier operators, restoration, and metrics
    io/               Image loading, numerical exports, and resource paths
    experiments/      Configurations, degradation, run records, and comparisons
    ui/               Windows, controls, views, and worker integration
    resources/        Small curated sample and reference package data
docs/                 Project instructions and research explanations
```

This is a proposed structure, not a requirement to create empty directories or placeholder files. A minimal set of maintained tests and build configuration will be added when needed. Generated outputs and dependency environments will not become source assets.

The UI will submit a validated configuration to a processing worker. The worker will call the numerical layer and report progress, results, and errors through Qt-safe signals. Each result will retain its inputs and configuration so subsequent edits cannot silently relabel an earlier output.

## Implementation stages

The stage identifiers below are planning labels, not existing GitHub issue numbers. I will create or reuse actual issues when implementation is requested.

### Stage 1: Establish the application foundation

Suggested issue title: `Initialize the desktop application foundation`

Suggested branch: `feat/app-foundation`

I will:

- Inspect the working tree, remote state, existing issues, and Git identity.
- Create a project-local `.venv`, verify interpreter compatibility, and install the required dependencies without changing global packages.
- Add packaging metadata, dependency constraints, and a documented development launch command.
- Extend `.gitignore` for environments, caches, and generated builds while preserving local exclusions.
- Import only the numerical source and small resources required for the next workflow, with source attribution.
- Preserve and run useful numerical checks before and after adaptation.
- Create a minimal PySide6 window that starts and closes cleanly.

Completion criteria:

- A fresh project environment can install the app dependencies and launch the window using the documented command.
- The numerical layer imports without Qt and its baseline checks pass.
- No code or resource path depends on a personal directory outside the repository.
- No application feature is marked complete merely because a placeholder window exists.

### Stage 2: Load images and complete the first RL workflow

Suggested issue title: `Restore an image with Richardson-Lucy`

Suggested branch: `feat/rl-restoration`

I will:

- Add image and PSF selection, image information, and a Gaussian PSF option.
- Validate supported dimensions, finite values, PSF normalization, and algorithm parameters.
- Start with scientific TIFF and PNG input; make any grayscale conversion or intensity conversion explicit.
- Add RL iteration controls and background processing without freezing the window.
- Display the input and restored image, with an explicit display range.
- Export the numerical result as float TIFF.
- Handle missing files, unsupported inputs, and processing failures with clear English messages.

Completion criteria:

- The complete load -> configure -> restore -> inspect -> save workflow works on a saved sample.
- The output matches the reused RL implementation on identical numerical inputs within an explicitly justified tolerance.
- UI controls and result labels remain consistent during and after a run.
- Saved output preserves the intended numerical values rather than saving a rendered screenshot.

### Stage 3: Add controlled experiments and method comparisons

Suggested issue title: `Compare restoration methods in controlled experiments`

Suggested branch: `feat/method-comparison`

I will:

- Add IF threshold and Tikhonov lambda controls and allow comparison with RL.
- Support a simulation mode with source image, Gaussian blur, photon level, and seed.
- Keep the same degraded image while restoration parameters change; provide a separate action to regenerate it.
- Allow optional ground truth for an already degraded image.
- Calculate MSE, PSNR, SSIM, and processing time with clearly recorded conventions.
- Add a comparison table, shared image views, and synchronized zoom/region inspection.
- Keep an in-memory history of meaningful runs and mark results from an earlier configuration clearly.

Completion criteria:

- All three methods operate on identical inputs for a comparison.
- Quality metrics are unavailable with an explanatory state when ground truth is missing or incompatible.
- Display adjustments do not change the measured numerical arrays.
- Timing identifies whether it covers the algorithm or the full operation; it excludes I/O and display work when presented as algorithm time.

### Stage 4: Explain RL and inspect iteration behavior

Suggested issue title: `Inspect Richardson-Lucy iterations and intermediate images`

Suggested branch: `feat/rl-inspector`

I will:

- Extend RL observation without changing the default update equation or reference configuration.
- Display the current estimate, blurred prediction, observation-to-prediction ratio, backprojected correction, and updated estimate for a selected iteration.
- Pair the intermediate images with formulas and short explanations.
- Plot PSNR/SSIM versus iteration count when ground truth exists.
- Use selected checkpoints and bounded storage rather than retaining every full-resolution intermediate array by default.

Completion criteria:

- Instrumented RL agrees with the baseline calculation on the same inputs.
- Intermediate images and metrics correspond to the selected iteration and the original run configuration.
- The research bars example demonstrates how later iterations can reduce a quality metric.
- Sweeps are described as exploratory analysis and do not silently replace fixed benchmark parameters.

### Stage 5: Reproduce and verify the research examples

Suggested issue title: `Verify restoration results against saved DeconvolutionLab2 references`

Suggested branch: `feat/reference-verification`

I will:

- Curate saved examples for cells, bars, and fibers, retaining their provenance and matching PSFs.
- Add the existing reference configuration: Gaussian sigma 2, photon level 200, seed 0, RL 5 iterations, and TRIF lambda 0.1.
- Read the saved float TIFF inputs rather than regenerate their noise for an exact report comparison.
- Show Python and reference outputs, numerical difference images, maximum absolute difference, and RMSE.
- Distinguish restoration quality against ground truth from agreement between implementations.
- Present ImageJ MSE/PSNR as ImageJ measurements and SSIM as a Python measurement.
- Label saved reference results clearly; do not imply that opening them reruns ImageJ.

Completion criteria:

- Six compatible RL/TRIF comparisons are reproducible from the saved inputs.
- The existing maximum absolute difference below 1e-6 is treated as a baseline for its recorded configuration, not a universal tolerance. Deviations are investigated and reported.
- Boundaries, initialization, normalization, epsilon, and output handling are documented alongside the comparison.
- The IF implementation is not presented as verified against a different plugin inverse-filter variant.
- No source research data or results are overwritten.

### Stage 6: Export reproducible experiments and complete the workflow

Suggested issue title: `Export reproducible restoration experiments`

Suggested branch: `feat/experiment-export`

I will:

- Export selected outputs as numerical TIFFs, measurements as CSV, and configurations as versioned JSON.
- Record input identities, relevant hashes, PSF settings, noise model, seed, parameters, boundary rules, intensity conventions, and environment versions.
- Make clear which files must accompany a JSON configuration; configuration alone does not contain the source pixels.
- Support reloading a configuration and locating its required inputs.
- Check invalid configurations, missing inputs, and existing destination files without silently losing results.

Completion criteria:

- An exported experiment can be reopened and rerun when its documented inputs are supplied.
- Exported TIFFs retain the numerical result and CSV values agree with the displayed measurements.
- Each export identifies its method and configuration unambiguously.
- Full export is completed here; the basic TIFF save action remains part of Stage 2.

### Stage 7: Package and document the Windows app

Suggested issue title: `Package and document the Windows desktop application`

Suggested branch: `build/windows-package`

I will:

- Choose a suitable packaging tool and keep only the build configuration needed to reproduce the desktop distribution.
- Verify the packaged app with bundled resources and paths containing spaces, from a directory outside the source checkout.
- Check operation without a separate server or network dependency.
- Document setup, controls, scientific assumptions, example reproduction, attribution, and limitations.
- Retain a small, useful demo image or recording for the README when available.
- Confirm licensing and third-party redistribution requirements before adding a license or bundling external binaries.
- Prepare release notes and distribution artifacts; publish a GitHub release only within the authorization given for that action.

Completion criteria:

- The Windows build completes the documented demo workflow without requiring an existing development Python environment.
- The source checkout has no disposable verification scripts, logs, intermediate builds, or redundant sample copies left behind.
- Installation and use instructions match what was actually verified; limitations of testing on other machines are stated.
- A release is described as published only after publication is confirmed.

## Scientific acceptance rules

- I will retain the established circular convolution and correctly centered, normalized PSF for baseline comparisons.
- I will preserve the definitions of thresholded IF, zeroth-order Tikhonov with L = I, and the reference RL initialization and epsilon.
- For the normalized research examples, PSNR uses L = 1. Other intensity ranges require an explicit, recorded convention.
- I will calculate metrics on original numerical values, never independently rescaled display images.
- Region zoom is for inspection; the standard reported measurements remain full-image measurements unless explicitly labeled otherwise.
- I will not assume that a sharper-looking image has better quantitative restoration quality.
- The original study contains 81 configurations derived from three source images. It is not a collection of 81 independent scenes.
- I will preserve the distinction between saved benchmark values, newly measured values, and platform-dependent regenerated data.
- I will describe OTF caching results relative to the measured baseline, not as evidence of superiority over unrelated optimized implementations.

## Development and GitHub workflow

For each independently reviewable change, I will:

1. Inspect the current repository and reuse an existing issue when appropriate.
2. Define the requirement and completion criteria in an English issue.
3. Work on a descriptive branch and make meaningful commits.
4. Run checks appropriate to the affected numerical or UI behavior.
5. Remove disposable verification artifacts before handoff and the final PR.
6. Open an English PR describing the problem, implemented behavior, validation, and remaining limitations.
7. Review the diff and address findings without presenting self-review as an independent review.
8. Merge or publish only within the authorization available for that action.
9. Update this plan's progress and relevant user documentation to reflect verified results.

The suggested stage branches are examples, not a quota of issues, commits, or PRs. I will combine or split work when its actual dependencies and reviewability justify doing so. Narrative descriptions will use first-person singular where natural.

## Validation and file retention

I will use existing numerical tests as the starting point, then add checks for real regressions, numerical correctness, and maintained development workflows. UI validation will cover the affected user flow and relevant failures. I will not add a new test framework or keep a script solely because it was useful once before handoff.

One-off scripts, debug logs, temporary images, scratch exports, and intermediate builds will be removed when they are no longer needed. I will keep only files with a clear role in application execution, development, maintenance, documentation, reproducible research, or requested deliverables. Ignoring an unnecessary file does not replace cleaning it up.

Cleanup will target exact, task-created paths and preserve existing user files, local configuration, hooks, and research references. I will summarize validation in the PR or handoff rather than preserve disposable artifacts as evidence.

## Progress

- [x] Record the agreed implementation plan.
- [x] Stage 1: Application foundation.
- [x] Stage 2: First RL restoration workflow.
- [x] Stage 3: Controlled experiments and method comparisons.
- [x] Stage 4: RL iteration inspection.
- [x] Stage 5: Saved reference verification.
- [x] Stage 6: Reproducible experiment export.
- [ ] Stage 7: Windows packaging and documentation.

Stage 1 implements [issue #1](https://github.com/BaoITuvagi/image-deconvolution/issues/1)
on `feat/app-foundation`. I created packaging metadata, pinned the verified runtime
dependencies, prepared a project-local `.venv`, adapted the numerical core with
attribution, retained two saved sample TIFFs with provenance hashes, and added a
minimal PySide6 window. The README documents setup, launch commands, and current
limitations. The resources live inside the Python package so they survive a
normal installation; the proposed `io/` and `experiments/` modules remain deferred.

Validation performed for Stage 1:

- Reran all five original numerical baseline checks before adaptation.
- Passed ten maintained regression checks after adaptation, including importing
  the core without Qt; `pip check` reported no broken requirements.
- Compared all imported numerical routines against the original source on even
  and odd rectangular arrays, plus RL on the saved cells input. Results matched
  exactly in this environment, including checkpoint and cache modes. This is not
  a universal cross-platform tolerance or a new ImageJ verification.
- Installed the project into a fresh temporary environment through the standard
  wheel build, then checked the numerical core, bundled sample hashes, and a Qt
  event loop that showed and closed the window from outside the checkout.
- Opened the documented module launcher on the Windows desktop, inspected its
  English text and layout, and closed it through the Close button with exit code 0.
  The installed desktop launcher also opened the same window and closed with Ctrl+Q.
- Removed the fresh verification environment, build output, generated source
  metadata, and bytecode caches after checking them. The project `.venv` remains
  available for development and is excluded from Git.

Stage 1 establishes the foundation only. Image import, user-input validation,
restoration controls, workers, result views, exports, ImageJ execution, and a
standalone Windows distribution are not implemented here. The existing research
package was read without modifying its files. No release is published by this stage.

Stage 1 PR #2 was self-reviewed, its ten checks rerun, and merged before Stage 2.
For multi-stage implementation assignments I validate UI, interactions, and
scientific workflows, self-review, merge when checks pass, and synchronize main
without pausing for routine acceptance. Releases still require separate scope.

Stage 2 (issue #3) adds grayscale TIFF/PNG loading with explicit integer conversion,
PSF generation or import with center-padding and unit normalization, validated RL
parameters, a background worker, synchronized image navigation, a shared display
range, and float32 TIFF save. Results retain immutable input and parameter snapshots;
edits mark an earlier result rather than relabel it. Closing during processing is
blocked until the worker finishes.

Stage 2 validation: all 15 numerical/I/O/desktop tests passed on Windows, including
invalid inputs, color/stack rejection, PSF failures, worker failures, close guards,
stale outputs, and numerical TIFF round trips. Native desktop inspection confirmed
sample loading and RL execution. Sandbox ACLs prevented temporary-file I/O in the
restricted process; the tests passed in the real Windows environment.
Disposable directories from those attempts were removed. The native save dialog
was opened and canceled; numerical save was exercised through the UI test.

Stages 3-4 (issue #5) share the immutable experiment snapshot and were implemented
together. Comparisons run IF, identity-regularized TRIF, and RL on one fixed
observation. Source simulation regenerates Poisson noise only through an explicit
action with recorded photon level, seed, intensity range, and input identities.
Optional ground truth is validated against shape, SSIM size, and declared range;
unavailable quality metrics show N/A and a reason. The four synchronized views
show the selected run's original input, truth, output, and a compatible history
output. History retains at most eight runs or 256 MiB, except a single larger run.

RL inspection retains up to eight five-image checkpoints within 64 MiB. It shows
the estimate, prediction, ratio, correction, and update with their equations and
iteration labels; PSNR and SSIM curves are separate selectable plots. Timing
includes OTF/algorithm work but excludes checkpoint copies, metrics, I/O, and
display. Captured and baseline RL agree exactly in the maintained regression.

All 21 checks passed, including fixed observations, seeded regeneration, metric
availability/range, checkpoint equations/bounds, comparison history, and worker
lifecycle. Native Windows inspection covered the bars comparison table and RL
inspector. On the saved bars input, sampled PSNR peaks near 16.03692 dB at iteration
74 and is 16.00629 dB at iteration 100 in this environment; this illustrates late
quality loss, not parameter selection for a benchmark. Newly required saved
ground truths and the bars input have provenance hashes. PR #4 completed Stage 2.

Stages 5-6 (issue #7) curate the fibers example, six saved DL2 TIFFs, and saved
ImageJ measurements/environment metadata with source hashes. New Python RL/TRIF
calculations on the saved inputs reproduce all six comparisons. The largest
absolute difference is 8.104919255735865e-7 for bars/RL, matching the recorded
configuration-specific baseline. The reference tab separates agreement from
ground-truth quality, labels ImageJ MSE/PSNR as saved, and labels SSIM as Python.
Native Windows inspection confirmed the table, matched image views, and signed
difference display. No ImageJ/JAR execution occurred.

Experiment export creates a new folder, refusing an existing destination. It
contains version-1 JSON, exact float64 .npy inputs, original source identities,
numerical/file hashes, method/PSF/noise settings, conventions, environment versions,
float32 numerical TIFF, CSV metrics, and available RL curve measurements. Reopen
checks conventions and hashes, permits locating moved input arrays, and restores
controls for an explicit rerun. JSON alone does not embed pixels. Repeated PSF
normalization on reload is checked with absolute output tolerance 1e-12 here;
this is separate from plugin agreement tolerance.

All 25 checks passed, including six references, exact input round trips, CSV/TIFF
agreement, UI export/reopen/rerun, missing/located/tampered arrays, invalid configs,
and overwrite refusal. Worker results are delivered through an explicitly queued
Qt slot before touching widgets. Source research remains unchanged. PR #6
completed Stages 3-4; packaging and release documentation follow separately.

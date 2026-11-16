# Image Deconvolution 0.1.0 — local Windows evaluation build

This build completes the seven planned desktop stages. It is prepared locally;
no GitHub Release is published.

The November 16, 2026 desktop review update corrects history selection, current-input
preview, ground-truth refresh, persistent configuration notices, RL quality labels,
and fitting across image tabs. I also consolidated saved examples, added compact
adaptive numeric controls and contextual availability, and made three restoration
outputs visible together with input quality and PSNR change. Preserved integer
images receive display suggestions without changing measurement pixels or L.
History clearing asks for confirmation; experiment export now selects a parent
folder and a new folder name. Reference provenance and extra columns are expandable.
The updated local folder is `dist/review-fixes/Image Deconvolution`; the original
evaluation build is retained separately.

Features include grayscale TIFF/PNG input with explicit intensity policies,
Gaussian/imported centered PSFs, worker-based IF/TRIF/RL restoration, explicit
seeded Poisson simulation, full-image quality metrics with compatible ground
truth, bounded history, synchronized zoom/pan, RL phase inspection and quality
curves, six saved DL2 reference comparisons, float TIFF save, and versioned
experiment export/reopen with exact input snapshots and integrity checks.

The distribution contains the bundled Python runtime, dynamic Qt Essentials
libraries, numerical dependencies, small saved research examples, documentation,
and third-party notices. Extract the complete folder and launch
`ImageDeconvolution.exe`. Source setup and build commands are in the README.

The low-level numerical conventions preserve the group's original research.
Saved DL2 agreement and saved ImageJ measurements do not imply a new plugin run.
No new restoration algorithm, real-image effectiveness, or superiority over
DeconvolutionLab2 is claimed. Parameter exploration is not benchmark tuning.

Known limits: Windows x64/Python 3.13 build target; unsigned folder distribution;
CPU processing; finite 2D grayscale images up to 16 megapixels; circular
boundaries; no cancellation for a running worker; in-memory history limits;
bounded RL capture; float32 TIFF rounding; no direct JAR invocation, color/3D,
blind deconvolution, or alternate boundary models. Another clean Windows machine
has not been tested. Research redistribution rights are not established by
attribution; keep this evaluation build local pending that confirmation.

Validation on Windows 11 x64 (build 26200), Python 3.13.6: all 32 numerical,
I/O and Qt workflow regression checks passed. The source and updated EXE passed
all ten desktop acceptance groups. The frozen app was tested from an empty
working directory, with PATH limited to Windows and Python environment variables
removed. Its built-in workflow
verification denied Python socket connections and exited with code 0. This does
not claim OS-level network isolation or testing on a separate clean machine.
Native inspection covered contextual controls, adaptive epsilon stepping,
three-method comparison, canceling history deletion, the actual parent-folder/name
export dialogs, JSON reopen and rerun. The 960 x 640 minimum layout was checked
with all five image views available. The six saved-reference comparisons reproduced maximum
absolute difference 8.104919255735865e-7; no ImageJ execution occurred.

Automated acceptance supplies dialog destinations while exercising real file
writes, validation, workers, views, and reruns. The separate native export test
used the real dialogs; its numerical TIFF, CSV baseline and PSNR change matched
the rerun. Qt test fixtures dispose windows on the GUI thread and yield the GIL
while waiting for workers, avoiding cleanup aborts and artificial worker starvation.

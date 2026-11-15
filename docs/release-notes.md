# Image Deconvolution 0.1.0 — local Windows evaluation build

This build completes the seven planned desktop stages. It is prepared locally;
no GitHub Release is published.

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

Validation on Windows 11 x64 (build 26200), Python 3.13.6: 26 regression checks
passed, with eight affected I/O/desktop checks rerun after the final UI fixes.
The source and final EXE passed all ten desktop acceptance groups. The frozen
folder was tested outside the checkout in a path with spaces, with PATH limited
to Windows and Python environment variables removed. Its built-in workflow
verification denied Python socket connections and exited with code 0. This does
not claim OS-level network isolation or testing on a separate clean machine.
Native inspection also covered sample load, three-method comparison and
synchronized zoom. The six saved-reference comparisons reproduced maximum
absolute difference 8.104919255735865e-7; no ImageJ execution occurred.

The final minimal-Qt build's native file dialog was opened and canceled. UI
export/reopen destinations in the automated acceptance run were supplied by the
driver; the file writes, validation, workers, views, and reruns were real.

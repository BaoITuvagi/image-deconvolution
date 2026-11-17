# Image Deconvolution 0.1.0 — local Windows evaluation build

This build completes the seven planned desktop stages. It is prepared locally;
no GitHub Release is published.

The input-workflow update puts each algorithm's full name directly in Method:
RL (Richardson-Lucy), IF (Thresholded inverse filtering), and TRIF (Zeroth-order
Tikhonov). The separate description row is removed; run and experiment-file
identifiers remain RL/IF/TRIF. PSF/intensity controls appear in Inputs only when
a simulation source exists, including after Generate, and remain available in
Restoration. Saved examples and degraded-image loading hide them in Inputs.
Generate preserves the current shared display range and source rendering;
changed simulation parameters affect the generated observation. Saved reference
comparisons retain their fixed RL/TRIF configuration. The new local folder is
`dist/input-workflow/Image Deconvolution`; earlier builds are preserved.

All 41 regression tests and eleven source/frozen desktop workflow groups passed.
Coverage includes all three method identifiers and experiment round trips,
source/degraded transitions, inline layout and full method labels at 9/12/16-point
fonts, and identical source pixels/rendering after changes to sigma, photon
level, seed, PSF mode/file and L, including a manual shared display range.
The EXE passed from an empty working directory without development Python on
PATH. Native Windows self-validation covered the saved-example layout, full
Method dropdown, actual source dialog, PSF/L visibility, L=2 generation with
retained source brightness and successful TRIF restoration. Ruff lint/format,
dependency checks and diff checks passed. No new ImageJ execution occurred.

The compact-PSF update uses an inline Open PSF filename field instead of a
button and separate summary. Click the field or press Enter/Space while it is
focused to choose a file. It is blank until a file is selected, and saved
examples show their bundled PSF filename. The shared row order is PSF mode,
Open PSF, Gaussian sigma, then intensity L in both Inputs and Restoration.
Gaussian mode disables Open PSF; Imported PSF disables Gaussian sigma. Switching
modes retains both values. Redundant PSF summaries, instructions below Generate,
and instructions in Image information are removed; factual metadata remains.
Numerical PSF preparation, explicit generation and saved presets are unchanged.
The new local folder is `dist/compact-psf/Image Deconvolution`; earlier builds
are preserved.

All 38 regression tests and eleven source/frozen workflow groups passed.
Checks cover inline layout at 9/12/16-point text sizes, mode and panel switches,
mouse/keyboard file selection, cancel/error retention, saved presets, explicit
generation, restoration and experiment round trips. The EXE passed from an empty
working directory with development Python removed from PATH. Native Windows
self-validation covered the actual PSF file dialog, filename display, disabled
fields under each mode, both panel placements and a successful RL restoration.
Ruff lint/format and dependency checks passed. No new ImageJ execution occurred.

The contextual-PSF update replaces the pinned sidebar group from the previous
build. PSF, sigma, Open PSF, intensity L and PSF information now appear as plain
rows only inside Inputs & simulation or Restoration. Inputs places them after
Noise seed and before Generate; Restoration places them before the algorithm
controls. L is immediately below Open PSF. Switching panels reuses the same
fields without changing values, observations or run history; Display & history
and Experiment files hide them. Saved reference presets and explicit generation
are retained. The updated local folder is
`dist/contextual-psf/Image Deconvolution`; earlier builds are preserved.

All 37 regression tests and eleven source/frozen desktop workflow groups passed.
Checks cover row placement and visibility, inline fields at 9/12/16-point text
sizes, repeated panel switches without changed configuration or observation,
saved presets, generation, restoration and experiment round trips. The EXE passed
from an empty working directory without development Python on PATH. Native
Windows self-inspection verified sigma/L edits in Inputs, retained values in
Restoration, hidden fields in Display/history and Experiment files, the actual
source file dialog, explicit Generate, and a successful restoration afterward.
Lint, formatting and dependency checks passed. No new ImageJ execution occurred.

The shared-controls update places PSF & intensity above the scrollable workflow
sections. Gaussian/imported PSF, sigma, Open PSF, PSF information and intensity L
remain accessible in both Inputs and Restoration. Editing them keeps the current
observation fixed until Generate is explicitly clicked; restoration uses the
currently selected PSF and L. Saved example now contains only Cells, Bars and
Fibers reference. Their IF threshold is corrected from 0.01 to 0.1; TRIF 0.1,
RL 5, epsilon 1e-6 and L=1 are retained. Load Bars reference and edit RL iterations
to explore longer runs. Existing input order, inline Load, retained Inputs panel
and explicit generation behavior are preserved. The updated local folder is
`dist/shared-psf/Image Deconvolution`; earlier builds are retained.

All 37 regression tests and eleven source/frozen desktop workflow groups passed.
Coverage includes all three preset configurations, pinned shared controls at
9/12/16-point text sizes, fixed observations after PSF/L edits, restoration with
the changed PSF, and immutable historical settings. The EXE passed from an empty
working directory with development Python removed from PATH. Native Windows
self-inspection verified the three-item selector, retained Inputs selection,
three-method reference comparison, the actual source file dialog, and changing
sigma/L and generating an observation without leaving Inputs. Ruff lint/format
and pip check passed. No new ImageJ execution occurred.

The source-generation update shows a newly opened clean source only in Current
ground truth. Current observation stays empty with a Generate prompt, and Run /
Compare stay disabled until generation succeeds. Gaussian PSFs use the source
dimensions before an observation exists; imported PSFs remain supported. Opening
another source clears the current observation/output preview and keeps immutable
run history. Saved examples and degraded-image loading still provide observations
immediately. The current local build is `dist/source-generation/Image Deconvolution`;
earlier builds are retained.

For this change, all 36 regression tests passed, including source-only previews,
Gaussian/imported PSFs before observation creation, restoration guards, failed
generation recovery, fixed observations, history selection, and loading another
source. Source and frozen builds passed all eleven desktop workflow groups. Native
Windows self-inspection exercised Open source through the real file dialog,
disabled restoration before Generate, observation creation, restoration, and
opening another source while preserving history. Ruff lint/format and pip check
passed. No ImageJ execution occurred.

The sidebar layout follow-up keeps field labels and their controls on one row,
with a trailing colon, across all sidebar sections. Loading policy and Saved
example precede the image-opening buttons. Photon level, seed, and Generate
follow those buttons; labeled image details and ground-truth status are grouped
at the bottom. The sidebar reserves room for fields and vertical scrollbars,
including larger text sizes. Dropdown choices and numerical conventions are
unchanged. The latest local folder is `dist/sidebar-layout/Image Deconvolution`;
earlier builds are retained.

For the sidebar follow-up, all 34 regression tests passed, including inline
fields and horizontal-scroll checks at 9/12/16-point text sizes and default/minimum
window layouts. Source and frozen builds passed all eleven desktop acceptance
groups; the EXE ran from an empty working directory without development Python
on PATH. Native Windows self-inspection confirmed the two loading choices,
saved-example loading, the bottom image-information group, and inline labels in
Inputs, Restoration, and Display & history. Ruff lint/format and pip check passed.
No ImageJ execution occurred.

The November 17, 2026 input-panel update places Load beside the saved-example
selector and keeps Inputs & simulation open after loading. Wrapped input status
messages now reserve their full height when their text, width, or font changes,
preventing clipping and overlap with the controls below. Both Loading policy
modes are retained. The updated folder is
`dist/input-panel-fixes/Image Deconvolution`; earlier builds are retained.

For this update, all 33 regression tests passed, including input layout checks at
9-, 12-, and 16-point text sizes and 960 x 640 / 1280 x 820 windows. Source and
frozen builds passed all eleven desktop acceptance groups. The EXE completed
verification from an empty working directory with the development Python
environment removed from PATH and Python environment variables unset. Native
Windows inspection confirmed inline Load, retained input-panel selection, and
fully visible status text after loading a saved example and opening a clean
source through the actual file dialog. Ruff lint/format and pip check passed.
These are self-validation results on the current machine; no ImageJ execution
occurred.

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

For the November 16 build, validation on Windows 11 x64 (build 26200), Python
3.13.6: all 32 numerical,
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

# Third-party notices and local distribution scope

I retain the research attribution in `attribution.md` and the resource manifest.
No project-wide license is assigned to the reused research. The local Windows
build is prepared for evaluation on this machine; public redistribution remains
subject to confirming the research code/data rights. No external ImageJ or
DeconvolutionLab2 binaries are included.

The build copies the installed dependency license texts and Python's license into
its `licenses/` directory, alongside the GNU LGPLv3 and GPLv3 license texts needed
for the LGPL notice. Those texts apply to their respective third-party components,
not to the whole application by implication.

| Component | Pinned version | License / source |
| --- | --- | --- |
| Python | 3.13.6 verified runtime | Python Software Foundation license; bundled `Python-LICENSE.txt`; https://www.python.org/downloads/source/ |
| NumPy | 2.3.3 | BSD and bundled component notices, including OpenBLAS where present; https://github.com/numpy/numpy/tree/v2.3.3 |
| SciPy | 1.16.2 | BSD and bundled component notices; https://github.com/scipy/scipy/tree/v1.16.2 |
| Pillow | 12.3.0 | MIT-CMU and bundled component notices; https://github.com/python-pillow/Pillow/tree/12.3.0 |
| PySide6 Essentials / Shiboken6 | 6.10.0 | Used under LGPLv3; https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.10.0 |
| Qt Core / Gui / Widgets / Network and required plugins | 6.10.0 | Used under LGPLv3 and their included third-party terms; https://download.qt.io/official_releases/qt/6.10/6.10.0/submodules/ |
| PyInstaller build tool / bootloader | 6.16.0 | GPL with bootloader distribution exception; https://pyinstaller.org/en/v6.16.0/license.html |

Qt DLLs remain separate and dynamically replaceable in `_internal/PySide6` and its
plugin directories; no license term here restricts reverse engineering needed to
debug modifications of LGPL libraries. The app does not modify Qt/PySide6. The
matching upstream source is available from the versioned source locations above.
Retain the notices and full folder structure with a permitted distribution.

Only the required Qt Widgets profile is bundled. The quality plot uses QPainter.
The spec filters optional image-format/input plugins and their QML, PDF, and
Virtual Keyboard DLLs. Qt Charts is excluded; its GPLv3/commercial terms differ
from the chosen LGPL components. Official references:

- https://doc.qt.io/qt-6.10/licensing.html
- https://doc.qt.io/qtforpython-6.10/licenses.html
- https://www.qt.io/development/open-source-lgpl-obligations
- https://doc.qt.io/qt-6.10/qtcharts-index.html

Windows system libraries remain subject to their respective vendor terms. The
build process also preserves the license files shipped by dependency wheels;
this document does not replace those complete notices or grant a new license to
third-party code. Confirm the complete redistribution scope before any public
release; no GitHub Release is created by the build command.

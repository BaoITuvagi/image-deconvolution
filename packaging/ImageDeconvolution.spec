"""PyInstaller folder build; keep Qt DLLs dynamically replaceable."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files("image_deconvolution", includes=["resources/**/*"])
for package in ["image-deconvolution", "numpy", "scipy", "Pillow", "PySide6-Essentials", "shiboken6"]:
    datas += copy_metadata(package)

a = Analysis(
    [str(root / "src/image_deconvolution/__main__.py")],
    pathex=[str(root / "src")], datas=datas,
    excludes=["PySide6.QtCharts", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtWebEngineCore",
              "PySide6.QtWebEngineWidgets", "tkinter", "matplotlib", "pandas", "IPython", "pytest"],
)
# Qt's GUI hook collects optional input/format plugins and their QML/PDF DLLs.
# This app draws numerical pixels directly and only needs the Widgets profile.
qt_libraries = {"Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "Qt6Network.dll", "Qt6OpenGL.dll"}
qt_plugins = {"platforms/qwindows.dll", "platforms/qoffscreen.dll", "styles/qmodernwindowsstyle.dll"}
def needed_binary(entry):
    destination = entry[0].replace("\\", "/")
    filename = Path(destination).name
    if destination.startswith("PySide6/plugins/"):
        return destination.removeprefix("PySide6/plugins/") in qt_plugins
    if filename.startswith("Qt6") and filename.endswith(".dll"):
        return filename in qt_libraries
    return True
a.binaries = [entry for entry in a.binaries if needed_binary(entry)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ImageDeconvolution",
          console=False, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Image Deconvolution")

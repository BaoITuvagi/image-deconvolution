"""Build the local Windows distribution with notices; do not publish it."""

import argparse
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path


def main():
    if sys.platform != "win32":
        raise SystemExit("Build this distribution on 64-bit Windows with Python 3.13.")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Parent directory for a fresh Image Deconvolution folder; existing builds are kept",
    )
    arguments = parser.parse_args()
    dist_root = (root / arguments.dist_dir).resolve()
    output = dist_root / "Image Deconvolution"
    if output.exists():
        raise SystemExit(
            "The distribution already exists. Preserve it or remove that exact task-created build before rebuilding."
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--distpath",
            str(dist_root),
            "--workpath",
            str(root / "build/windows"),
            str(root / "packaging/ImageDeconvolution.spec"),
        ],
        cwd=root,
        check=True,
    )
    for name in [
        "README.md",
        "docs/user-guide.md",
        "docs/release-notes.md",
        "docs/attribution.md",
        "docs/third-party-notices.md",
        "docs/plan.md",
    ]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    if (root / "docs/images/workspace.png").is_file():
        (output / "docs/images").mkdir()
        shutil.copyfile(root / "docs/images/workspace.png", output / "docs/images/workspace.png")
    licenses = output / "licenses"
    licenses.mkdir()
    for package in [
        "numpy",
        "scipy",
        "Pillow",
        "PySide6-Essentials",
        "shiboken6",
        "pyinstaller",
    ]:
        d = distribution(package)
        for item in d.files or []:
            if any(
                token in item.name.lower() for token in ["license", "copying", "notice"]
            ) and item.suffix.lower() in {".txt", ".md", ""}:
                source = Path(d.locate_file(item))
                if source.is_file():
                    relative = Path(package) / str(item).replace("..", "_")
                    target = licenses / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    shutil.copyfile(Path(sys.base_prefix) / "LICENSE.txt", licenses / "Python-LICENSE.txt")
    for source in (root / "third_party/licenses").glob("*.txt"):
        shutil.copyfile(source, licenses / source.name)
    for pattern in [
        "*Charts*",
        "*VirtualKeyboard*",
        "*virtualkeyboard*",
        "Qt6Qml*",
        "Qt6Quick*",
        "Qt6Pdf*",
    ]:
        if list(output.rglob(pattern)):
            raise SystemExit(f"Unexpected optional Qt component found ({pattern}); investigate this build.")
    print(f"Built local distribution: {output}")


if __name__ == "__main__":
    main()

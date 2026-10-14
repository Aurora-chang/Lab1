"""Build the source upload archive for Lab1_Colab.ipynb."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

FILES = ["dataset.py", "model.py", "metrics.py", "plots.py", "train.py", "inference.py",
         "compare.py", "download_dataset.py", "requirements.txt", "README.md", "report_template.md"]


def main():
    root = Path(__file__).resolve().parent
    archive = root / "Lab1_colab_code.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name in FILES:
            bundle.write(root / name, arcname=name)
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

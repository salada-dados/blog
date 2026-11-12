"""Download the Central Bank of Brazil's Monetary Policy Report, September 2026 (pt and en).

Usage: python download_rpm.py
Writes data/rpm202609p.pdf (Portuguese) and data/rpm202609i.pdf (English).

Source: Banco Central do Brasil. Reproduction permitted only if source is stated as follows:
Monetary Policy Report, volume 2, nr. 3. The PDFs are downloaded here, not stored in this repository.
"""
import hashlib
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
FILES = {
    "rpm202609p.pdf": (
        "https://www.bcb.gov.br/content/ri/relatorioinflacao/202609/rpm202609p.pdf",
        "274c8ac6e603703dcb3d73dd758302cd10b8056580e4c44966773fab05db066b",
    ),
    "rpm202609i.pdf": (
        "https://www.bcb.gov.br/content/ri/inflationreport/202609/rpm202609i.pdf",
        "f9ed8609a7c5fda63ff22bdbbc9cddc37f22c9bb621f6b172a2b9af01316081a",
    ),
}

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, (url, sha256) in FILES.items():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise SystemExit(f"{url} did not return a PDF")
        (DATA / name).write_bytes(response.content)
        digest = hashlib.sha256(response.content).hexdigest()
        # The Central Bank sometimes republishes a report with fixes. The code still runs on a
        # new file, but page numbers and answers in questions.json may no longer match.
        note = "same file as the article" if digest == sha256 else "WARNING: differs from the file used in the article"
        print(f"{name}: {len(response.content) / 1e6:.1f} MB, {note}")

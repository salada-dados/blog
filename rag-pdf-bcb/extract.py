"""Turn each PDF into Markdown with three extractors and time each one.

Usage: python extract.py [pypdf|pymupdf4llm|docling ...]
Writes out/<extractor>_<lang>.md (one "<!-- page N -->" marker per page) and out/timings.json.
Docling also saves the crop of every figure it finds (out/figures_<lang>/) for describe.py.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
PDFS = {"pt": HERE / "data/rpm202609p.pdf", "en": HERE / "data/rpm202609i.pdf"}


def with_pypdf(pdf: Path, lang: str) -> str:
    from pypdf import PdfReader
    pages = PdfReader(pdf).pages
    return "\n\n".join(f"<!-- page {i} -->\n{p.extract_text()}" for i, p in enumerate(pages, 1))


def with_pymupdf4llm(pdf: Path, lang: str) -> str:
    import pymupdf4llm
    chunks = pymupdf4llm.to_markdown(str(pdf), page_chunks=True)
    return "\n\n".join(f"<!-- page {i} -->\n{c['text']}" for i, c in enumerate(chunks, 1))


def with_docling(pdf: Path, lang: str) -> str:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # the PDF already has a text layer; OCR would only add time and errors
    opts.do_table_structure = True
    opts.generate_picture_images = True
    opts.images_scale = 2.0  # crops at ~144 dpi, so the vision model can read axis labels
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = converter.convert(pdf).document
    doc.save_as_json(OUT / f"docling_{lang}.json")

    figures = OUT / f"figures_{lang}"
    figures.mkdir(exist_ok=True)
    n = 0
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem) and (img := item.get_image(doc)) is not None:
            img.save(figures / f"{item.self_ref.split('/')[-1]}.png")
            n += 1
    print(f"  {n} figures saved")

    parts = []
    for page in sorted(doc.pages):
        parts.append(f"<!-- page {page} -->\n" + doc.export_to_markdown(page_no=page))
    return "\n\n".join(parts)


EXTRACTORS = {"pypdf": with_pypdf, "pymupdf4llm": with_pymupdf4llm, "docling": with_docling}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    timings_file = OUT / "timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else {}
    for name in sys.argv[1:] or EXTRACTORS:
        for lang, pdf in PDFS.items():
            start = time.perf_counter()
            md = EXTRACTORS[name](pdf, lang)
            timings[f"{name}/{lang}"] = round(time.perf_counter() - start, 1)
            (OUT / f"{name}_{lang}.md").write_text(md, encoding="utf-8")
            print(f"{name}/{lang}: {timings[f'{name}/{lang}']}s, {len(md):,} chars")
            timings_file.write_text(json.dumps(timings, indent=1))

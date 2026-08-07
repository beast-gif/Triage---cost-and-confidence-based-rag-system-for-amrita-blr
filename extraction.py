# run_extraction.py
import json
from pathlib import Path
from extractor import extract_chunks

def process_saved_html(raw_dir: str = "raw_html", out_file: str = "chunks.jsonl"):
    raw_path = Path(raw_dir)
    all_chunks = []

    html_files = list(raw_path.glob("*.html"))
    print(f"Found {len(html_files)} scraped pages")

    for html_file in html_files:
        meta_file = html_file.with_suffix("").with_suffix(".meta.json")
        # meta files were saved as {name}.meta.json, html as {name}.html
        meta_file = raw_path / f"{html_file.stem}.meta.json"

        if not meta_file.exists():
            print(f"[skip] no metadata for {html_file.name}")
            continue

        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        chunks = extract_chunks(
            html=html,
            source_url=meta["url"],
            page_title=meta["title"],
        )
        all_chunks.extend(chunks)
        print(f"[ok] {meta['url']} -> {len(chunks)} chunks")

    # save as JSONL — one chunk per line, easy to stream into embedder
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks written to {out_file}")
    return all_chunks


if __name__ == "__main__":
    process_saved_html()
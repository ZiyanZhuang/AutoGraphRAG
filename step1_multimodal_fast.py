#!/usr/bin/env python3
"""
Multimodal PDF Parser for GraphRAG Pipeline
--------------------------------------------
Extracts text and images from PDF files, producing enhanced text files with image placeholders.
Supports batch processing of multiple PDFs from a folder or a single PDF via command-line arguments.

Usage:
    # Process all PDFs in the default raw_pdfs folder under project root
    python step1_multimodal_fast.py --project-root /path/to/project

    # Process a single PDF file (will be copied to raw_pdfs/ before processing)
    python step1_multimodal_fast.py --project-root /path/to/project --pdf /path/to/document.pdf

Author: AutoGraphRAG Team
Date: 2026-02-22
Version: 2.0.0
"""

import fitz  # PyMuPDF
import os
import hashlib
import time
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Global Configuration
CHUNK_SIZE = 10  # Pages per process


def get_file_hash(content: bytes) -> str:
    """Return MD5 hash of binary content."""
    return hashlib.md5(content).hexdigest()


def process_page_chunk(
    pdf_path: Path,
    project_root: Path,
    page_start: int,
    page_end: int,
    image_output_dir: Path,
):
    """
    Worker function to process a range of pages.
    Each process opens its own handle to the PDF.
    Returns list of (page_num, text_content) tuples.
    """
    try:
        doc = fitz.open(pdf_path)
        paper_id = pdf_path.stem
        chunk_results = []

        for page_num in range(page_start, min(page_end, len(doc))):
            page_id = page_num + 1
            page = doc[page_num]

            # --- 1. Extract text blocks ---
            text_blocks = page.get_text(
                "blocks"
            )  # (x0,y0,x1,y1,"text",block_no,block_type)

            # --- 2. Extract images ---
            image_list = page.get_images(full=True)

            mixed_blocks = []

            # Add text blocks
            for block in text_blocks:
                if block[4].strip():
                    mixed_blocks.append(
                        {
                            "type": "text",
                            "bbox": fitz.Rect(block[:4]),
                            "content": block[4],
                            "y0": block[1],
                            "x0": block[0],
                        }
                    )

            # Process images
            for img in image_list:
                xref = img[0]
                try:
                    rects = page.get_image_rects(xref)
                    for rect in rects:
                        # Skip tiny images (likely icons/lines)
                        if rect.width < 50 or rect.height < 50:
                            continue

                        # Extract image bytes
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        img_hash = get_file_hash(image_bytes)
                        image_filename = f"{img_hash}.{image_ext}"
                        image_path = image_output_dir / image_filename

                        # Save image if not already present
                        if not image_path.exists():
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                        # Use project-relative path for the Markdown link
                        rel_path = image_path.relative_to(project_root)

                        mixed_blocks.append(
                            {
                                "type": "image",
                                "bbox": rect,
                                "content": f"![Figure]({str(rel_path)})",
                                "image_path": str(rel_path),
                                "y0": rect.y0,
                                "x0": rect.x0,
                            }
                        )
                except Exception as e:
                    # Failsafe for corrupt images
                    pass

            # --- 3. Sort blocks by vertical then horizontal position ---
            mixed_blocks.sort(key=lambda b: (b["y0"], b["x0"]))

            # --- 4. Assemble page content ---
            page_content = []
            for block in mixed_blocks:
                if block["type"] == "text":
                    text = block["content"].strip()
                    annotated_text = f"[[PAPER: {paper_id} | PAGE: {page_id}]] {text}\n"
                    page_content.append(annotated_text)
                elif block["type"] == "image":
                    placeholder = (
                        f"\n[[FIGURE_REF: {block['image_path']} | PAGE: {page_id}]]\n"
                    )
                    page_content.append(placeholder)

            chunk_results.append((page_num, "".join(page_content)))

        doc.close()
        return chunk_results
    except Exception as e:
        print(f"Error in worker process for pages {page_start}-{page_end}: {e}")
        return []


def process_pdf(pdf_path: Path, project_root: Path):
    """Process a single PDF file."""
    start_time = time.time()

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    print(
        f"Processing {pdf_path.name} ({total_pages} pages) with {os.cpu_count()} cores..."
    )

    # Prepare output directories
    image_dir = project_root / "assets" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir = project_root / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Divide work into chunks
    chunks = []
    for i in range(0, total_pages, CHUNK_SIZE):
        chunks.append((i, min(i + CHUNK_SIZE, total_pages)))

    results = []

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_page_chunk, pdf_path, project_root, start, end, image_dir
            ): (start, end)
            for start, end in chunks
        }
        for future in as_completed(futures):
            chunk_data = future.result()
            results.extend(chunk_data)

    # Reassemble in correct order
    results.sort(key=lambda x: x[0])
    final_text = [content for _, content in results]

    # Save output
    paper_id = pdf_path.stem
    output_file = output_dir / f"{paper_id}_processed.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(final_text))

    duration = time.time() - start_time
    print(
        f"Done! Processed {total_pages} pages in {duration:.2f}s ({total_pages / duration:.2f} pages/sec)"
    )
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and images from PDFs for GraphRAG."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Project root directory. All outputs will be placed relative to this path.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Optional: Path to a single PDF file to process. If not provided, processes all PDFs in <project-root>/raw_pdfs.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    print(f"Project root: {project_root}")

    # Determine which PDFs to process
    if args.pdf:
        single_pdf = Path(args.pdf).resolve()
        if not single_pdf.exists():
            print(f"Error: Specified PDF file not found: {single_pdf}")
            return
        # Ensure raw_pdfs directory exists and copy the PDF there
        raw_pdfs_dir = project_root / "raw_pdfs"
        raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        dest_pdf = raw_pdfs_dir / single_pdf.name
        shutil.copy2(single_pdf, dest_pdf)
        print(f"Copied {single_pdf.name} to {dest_pdf}")
        pdf_files = [dest_pdf]
    else:
        raw_pdfs_dir = project_root / "raw_pdfs"
        if not raw_pdfs_dir.exists():
            print(
                f"Error: raw_pdfs directory not found at {raw_pdfs_dir}. Please create it and place PDF files there, or use --pdf to specify a file."
            )
            return
        pdf_files = list(raw_pdfs_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {raw_pdfs_dir}")
            return
        print(f"Found {len(pdf_files)} PDF(s) in {raw_pdfs_dir}")

    total_start = time.time()
    for pdf_file in pdf_files:
        process_pdf(pdf_file, project_root)
    print(f"Total batch time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

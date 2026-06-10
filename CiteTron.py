# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:52:25 2026

@author: milab
"""

#%%

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
from collections import defaultdict
import pdfplumber
from pypdf import PdfReader
from fastembed import TextEmbedding
import faiss
import numpy as np
import nltk
from groq import Groq
from dotenv import load_dotenv

nltk.download("punkt_tab", quiet=True)

load_dotenv(Path(__file__).parent / ".env")

#%%
@dataclass
class TextChunk:
    source: str          # PDF filename (stem)
    source_path: str     # full path as string
    chunk_index: int     # 0-based position within the document
    text: str            # chunk content
    char_start: int      # start offset in the full document text
    char_end: int        # end offset in the full document text
    metadata: dict = field(default_factory=dict)  # extendable (page range, etc.)
 


def extract_text_from_pdf(path: Path, fast: bool = False) -> str:
    """
    Extract all text from a PDF file.
 
    Args:
        path: path to the PDF file
        fast: if True, use pypdf (much faster, good for prose/academic papers);
              if False (default), use pdfplumber (slower but handles tables
              and multi-column layouts more accurately).
 
    Falls back gracefully if a page fails to parse.
    Returns an empty string for password-protected or image-only PDFs.
    """
    pages: list[str] = []
 
    if fast:
        try:
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    pages.append(text)
                except Exception as exc:
                    print(f"  [warn] page {page_num} of '{path.name}' skipped: {exc}")
        except Exception as exc:
            print(f"[error] could not open '{path.name}': {exc}")
            return ""
    else:
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        pages.append(text)
                    except Exception as exc:
                        print(f"  [warn] page {page_num} of '{path.name}' skipped: {exc}")
        except Exception as exc:
            print(f"[error] could not open '{path.name}': {exc}")
            return ""
 
    return "\n".join(pages)

 
#%%
def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove common PDF artefacts.
    Keeps paragraph breaks (double newlines) intact.
    """
    # Collapse runs of spaces/tabs within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


#%%

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 2,
) -> Generator[tuple[int, int, str], None, None]:
    """
    Yield (char_start, char_end, chunk_text) tuples using sentence-boundary chunking.

    Groups sentences into chunks that stay under chunk_size characters.
    The overlap parameter controls how many sentences carry over between chunks,
    preserving context across boundaries without cutting mid-argument.

    Args:
        text:       full document text
        chunk_size: max characters per chunk (default 512)
        overlap:    number of sentences to carry over between chunks (default 2)
    """
    if not text:
        return

    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return

    # Build a char-offset map for each sentence so we can report char_start/char_end
    offsets: list[int] = []
    pos = 0
    for sent in sentences:
        idx = text.find(sent, pos)
        offsets.append(idx if idx != -1 else pos)
        pos = offsets[-1] + len(sent)

    i = 0
    while i < len(sentences):
        chunk_sents: list[str] = []
        chunk_len = 0
        j = i
        while j < len(sentences):
            s = sentences[j]
            if chunk_sents and chunk_len + len(s) + 1 > chunk_size:
                break
            chunk_sents.append(s)
            chunk_len += len(s) + 1
            j += 1

        chunk = " ".join(chunk_sents).strip()
        if chunk:
            char_start = offsets[i]
            char_end = offsets[j - 1] + len(sentences[j - 1])
            yield char_start, char_end, chunk

        # Advance, carrying over `overlap` sentences for context continuity
        i = max(i + 1, j - overlap)


#%%

def load_and_chunk_pdfs(
    directory: str | Path,
    chunk_size: int = 512,
    overlap: int = 2,
    recursive: bool = False,
    fast: bool = False,
) -> list[TextChunk]:
    """
    Find all PDFs in *directory*, extract their text, and return a flat list
    of TextChunk objects ready for embedding or further processing.
 
    Args:
        directory:  path to the folder containing the PDFs
        chunk_size: max characters per chunk (default 512)
        overlap:    sentences carried over between chunks (default 2)
        recursive:  if True, also search subdirectories
        fast:       if True, use pypdf backend (much faster, good for prose);
                    if False (default), use pdfplumber (better for tables/columns)
 
    Returns:
        List of TextChunk dataclass instances.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
 
    glob_pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(root.glob(glob_pattern))
 
    if not pdf_paths:
        print(f"[warn] no PDF files found in '{root}'")
        return []
 
    backend = "pypdf" if fast else "pdfplumber"
    print(f"Found {len(pdf_paths)} PDF(s) in '{root}' [backend: {backend}]")
 
    all_chunks: list[TextChunk] = []
 
    for pdf_path in pdf_paths:
        print(f"  Processing: {pdf_path.name}")
 
        raw_text = extract_text_from_pdf(pdf_path, fast=fast)
        if not raw_text.strip():
            print(f"  [warn] no text extracted from '{pdf_path.name}' — skipping")
            continue
 
        clean = clean_text(raw_text)
        doc_chunks = list(chunk_text(clean, chunk_size=chunk_size, overlap=overlap))
 
        print(f"    → {len(doc_chunks)} chunks ({len(clean):,} chars total)")
 
        for idx, (char_start, char_end, text) in enumerate(doc_chunks):
            all_chunks.append(
                TextChunk(
                    source=pdf_path.stem,
                    source_path=str(pdf_path),
                    chunk_index=idx,
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
 
    print(f"\nTotal chunks produced: {len(all_chunks)}")
    return all_chunks

#%%

FOLDER = Path(__file__).parent / "Files here"
FOLDER.mkdir(exist_ok=True)
 
chunks = load_and_chunk_pdfs(FOLDER, chunk_size=512, overlap=2, fast=False)
 
# Preview first 3 chunks
for chunk in chunks[:3]:
    print(f"\n[{chunk.source}] chunk #{chunk.chunk_index}")
    print(f"  chars {chunk.char_start}–{chunk.char_end}")
    print(f"  {chunk.text[:120]}...")

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed([chunk.text for chunk in chunks]))


# Stack all embeddings into a matrix (n_chunks x 384)
vectors = np.array(embeddings, dtype="float32")

# L2 index (euclidean distance) — for cosine similarity, normalize first
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(384)  # Inner product = cosine similarity after normalization
index.add(vectors)


#%%
query = "In the field of econometrics, these findings of non-Poisson inter-arrival times have not led to any major movements towards modeling financial data that arrive at uneven intervals."
query_vec = np.array(list(model.embed([query])), dtype="float32")
faiss.normalize_L2(query_vec)


k = 5  # how many chunks to retrieve
scores, indices = index.search(query_vec, k)

top_chunks = [chunks[i] for i in indices[0]]



for score, idx in zip(scores[0], indices[0]):
    print("=" * 80)
    print("Score:", score)
    print("Source:", chunks[idx].source)
    print(chunks[idx].text[:500])

#%%


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def classify_stance(claim: str, chunk_text: str, chunk: object) -> dict:
    prompt = f"""You are a scientific literature analyst. Your job is to determine whether a text excerpt supports, contradicts, or is neutral toward a given claim.

CLAIM: {claim}

EXCERPT (from paper "{chunk.source}", chunk #{chunk.chunk_index}):
{chunk_text}

Analyze the excerpt carefully. Consider:
- Direct statements that confirm or deny the claim (SUPPORTS)
- Implicit agreement or disagreement through context (CONTRADICTS)
- Whether the excerpt is simply on the same topic without taking a stance (NEUTRAL)
- Whether the excerpt simply has nothing to do with the stance (ERRONEOUS)

Respond in exactly this format:
LABEL: <SUPPORTS|CONTRADICTS|NEUTRAL|ERRONEOUS>
CONFIDENCE: <0.0-1.0>
REASON: <one sentence explaining your judgment>"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic — you want consistent classification
        max_tokens=100,
    )
    
    text = response.choices[0].message.content.strip()
    
    result = {"source": chunk.source, "chunk_index": chunk.chunk_index, "chunk_text": chunk_text}
    for line in text.splitlines():
        if line.startswith("LABEL:"):
            result["label"] = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["confidence"] = 0.0
        elif line.startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()
    return result
#%%
# Run — group results by source paper
print(f"CLAIM: {query}\n")

results_by_paper: dict[str, list[dict]] = defaultdict(list)
for score, chunk in zip(scores[0], top_chunks):
    stance = classify_stance(query, chunk.text, chunk)
    stance["retrieval_score"] = score
    results_by_paper[stance["source"]].append(stance)

for paper, stances in results_by_paper.items():
    print("=" * 80)
    print(f"PAPER: {paper}  ({len(stances)} chunk(s) retrieved)")
    print("-" * 80)
    for s in stances:
        print(f"  chunk #{s['chunk_index']:>3}  [{s.get('label','?'):<12}]  "
              f"confidence={s.get('confidence','?')}  retrieval={s['retrieval_score']:.3f}")
        print(f"  Reason: {s.get('reason','')}")
        print(f"  Text:   {s['chunk_text'][:250]}...\n")







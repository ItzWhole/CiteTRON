# CiteTron

A lightweight RAG pipeline that ingests a folder of academic PDFs and classifies whether a given claim is **supported**, **contradicted**, or **neutral** based on the literature.

Built without heavy dependencies — no PyTorch, no LangChain, no boilerplate.

---

## How it works

```
PDFs → text extraction → chunking → embeddings → FAISS index → retrieval → LLM stance classification
```

1. **Ingestion** — extracts and cleans text from all PDFs in the `Files here/` folder using `pdfplumber` (accurate) or `pypdf` (fast)
2. **Chunking** — splits documents into sentence-boundary chunks with sentence-level overlap, preserving argument coherencegit add "Files here/.gitkeep"
3. **Embedding** — encodes every chunk into a 384-dimensional vector using `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, no torch)
4. **Indexing** — stores vectors in a FAISS flat index with cosine similarity
5. **Retrieval** — embeds the input claim, fetches the top-k most semantically similar chunks
6. **Stance detection** — passes each retrieved chunk to an LLM (Groq / Llama 3.1) and classifies it as `SUPPORTS`, `CONTRADICTS`, `NEUTRAL`, or `ERRONEOUS`

---

## Stack

| Component | Library |
|---|---|
| PDF extraction | `pdfplumber`, `pypdf` |
| Embeddings | `fastembed` (BAAI/bge-small-en-v1.5) |
| Vector search | `faiss-cpu` |
| Stance classification | `groq` (llama-3.1-8b-instant) |

No PyTorch. No LangChain. Runs on CPU.

---

## Installation

```bash
pip install pdfplumber pypdf fastembed faiss-cpu groq python-dotenv nltk
```

---

## Setup

1. Get a free API key from [Groq](https://console.groq.com)

2. Copy `.env.example` to `.env` and paste your key in:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_groq_api_key_here
```

> `.env` is gitignored and will never be committed — your key stays local.

3. Drop your PDF files into the `Files here/` folder next to `CiteTron.py`

4. Set your claim in the `query` variable and run the script

---

## Usage

Edit the `query` variable near the bottom of `CiteTron.py`:

```python
query = "Wealth distribution follows a power law."
```

Then run the script. Results are grouped by paper:

```
CLAIM: Wealth distribution follows a power law.

================================================================================
PAPER: econoc  (2 chunk(s) retrieved)
--------------------------------------------------------------------------------
  chunk # 47  [SUPPORTS    ]  confidence=0.93  retrieval=0.871
  Reason: The excerpt explicitly discusses Pareto-distributed wealth and its empirical validation.
  Text:   ...the upper tail of the wealth distribution is well described by a power law
          with exponent α ≈ 1.4, consistent across multiple economies...

  chunk # 12  [NEUTRAL     ]  confidence=0.81  retrieval=0.743
  Reason: The excerpt discusses income inequality metrics but does not address power laws directly.
  Text:   ...Gini coefficients have risen steadily since the 1980s in most OECD countries...
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `512` | Max characters per chunk |
| `overlap` | `2` | Sentences carried over between consecutive chunks |
| `fast` | `False` | Use `pypdf` instead of `pdfplumber` |
| `k` | `5` | Number of chunks to retrieve per query |
| `model` | `llama-3.1-8b-instant` | Groq model for stance classification |
| `temperature` | `0.0` | LLM temperature (0 = deterministic) |

---

## Design decisions

**Why not LangChain?** — The pipeline is simple enough to reason about directly. Fewer abstractions means easier debugging and full control over chunking, retrieval, and prompting.

**Why fastembed over sentence-transformers?** — fastembed runs on ONNX and has no PyTorch dependency, making it installable anywhere without GPU or CUDA setup.

**Why FAISS over a vector database?** — For a folder of papers, an in-memory flat index is instant to build and query. A server-based vector DB would add operational overhead with no benefit at this scale.

**Why sentence-boundary chunking?** — Character-based chunking can cut mid-argument, degrading both retrieval scores and stance accuracy. Splitting on sentence boundaries keeps semantic units intact, and sentence-level overlap preserves context across chunk boundaries.

**Why an LLM for stance instead of a local NLI model?** — Small cross-encoders (DeBERTa-based) classify entailment well on clean sentence pairs but struggle with the hedged, citation-dense language of academic papers. An LLM handles implicit contradiction and context-dependent support more reliably.

---

## Limitations

- Equations and mathematical notation extracted from PDFs are often garbled — semantic meaning of math-heavy passages may be partially lost
- Stance classification is sensitive to chunk boundaries; a key sentence split across two chunks may reduce accuracy
- Performance degrades on image-only or scanned PDFs with no text layer

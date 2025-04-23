# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

CONCORDIA compares two functional-annotation sources—old vs new, RAST vs UniProt, manual vs AI—and streams a tidy file with **cosine similarity** *and* **LLM labels**.

---

## Installation
```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install          # deps + CLI
poetry shell            # activate venv
```

---

## Quick-start recipes

| Purpose | Command |
|---------|---------|
| CSV with **default LLM** (o3-mini → apps-dev) | `concord example_data/annotations_test.csv` |
| TSV with **GPT-4o** (prod) | `concord example_data/annotations_test.tsv --llm-model gpt4o` |
| **Embed-only**, no LLM | `concord …csv --mode local` |
| **Hybrid** (embed first, LLM only for 0.30 ≤ sim ≤ 0.85) | `concord …csv --mode hybrid` |
| **Dual** (always embed **and** always LLM) | `concord …csv --mode dual` |
| Ad-hoc strings | `concord --text-a "RecA" --text-b "DNA recombinase A"` |

---

## Accepted input formats

| Extension | Loader behaviour | Note |
|-----------|------------------|------|
| **`.csv`** | `pandas.read_csv(sep=",")` | default |
| **`.tsv` / `.tab`** | `pandas.read_csv(sep="\t")` | or `--sep "\t"` |
| **`.json`** | `pandas.read_json()` | list-of-objects **or** column-orient |

The first two textual columns that **don’t** end with **`id`** are selected unless you pass `--col-a / --col-b`.

---

## Minimal sample CSV
```csv
annotation_a,annotation_b
DNA repair protein RecA,Recombinase A
Hypothetical protein,Uncharacterized protein
```
Any extra columns (e.g. `gene_id`) are preserved in the output.

---

## CLI options

| Flag | Description |
|------|-------------|
| **`FILE`** | Input table (`.csv`, `.tsv`, `.json`). |
| `--text-a / --text-b` | Compare two strings instead of a file. |
| `--mode` | `llm` (default) | `local` | `hybrid` | `dual` |
| `--llm-model` | Gateway LLM (`gpto3mini`, `gpt4o`, …). |
| `--output` | Destination path (file-mode only). |
| `--cfg` | Alternate YAML config (default `concord/config.yaml`). |
| `--col-a / --col-b` | Explicit annotation columns. |
| `--sep` | Custom delimiter for text files. |

---

## Modes

| Mode | What happens | Extra columns |
|------|--------------|---------------|
| **llm** | Skip embeddings; **every pair → LLM** | `label`, `note` |
| **local** | PubMedBERT embeddings → cosine → heuristic label | `similarity`, `label` |
| **hybrid** | Cosine first; LLM only when 0.30 ≤ sim ≤ 0.85 | `similarity`, `label`, `note?` |
| **dual** | Cosine label **and** LLM label for every row | `similarity`, `cosine_label`, `label`, `note` |

> **Embedding model** `NeuML/pubmedbert-base-embeddings` (Apache-2.0).

---

## Output columns

| Name | Description |
|------|-------------|
| *(all originals)* | Copied unchanged |
| `similarity` | Cosine similarity (null in pure LLM) |
| `cosine_label` | Heuristic label (dual mode only) |
| `label` | Final label (LLM or heuristic) |
| `note` | Very short LLM reason (blank if heuristic) |

### Label definitions

| Label | Meaning |
|-------|---------|
| **Identical** | Same function; minor wording difference |
| **Synonym**  | Same meaning via alternative phrasing |
| **Partial**  | Overlap; one broader/narrower |
| **New**      | Unrelated functions |
| **Unknown**  | LLM reply empty / unparseable |
| **Error**    | Exception captured; see `note` |

---

## Prompting — how to tweak

All prompt logic lives in **`concord/llm/prompts.py`**:

```python
LABEL_SET = ["Identical", "Synonym", "Partial", "New"]

_FEW_SHOT = """\
A: ATP synthase subunit beta
B: ATP synthase β subunit
Identical — wording difference only
"""

def build_annotation_prompt(a: str, b: str) -> str:
    return (
        f"{_FEW_SHOT}\n\n"
        f"Classify the relationship.\nA: {a}\nB: {b}\n\n"
        "Respond: <Label> — <very short reason>.\n"
        f"Allowed labels: {', '.join(LABEL_SET)}."
    )
```

* Add few-shot examples, chain-of-thought, or demand JSON—edit this file only.

---

## Config snapshot (`concord/config.yaml`)

```yaml
engine:
  mode: llm
  hybrid_thresholds: {lower: 0.30, upper: 0.85}

llm:
  model: gpto3mini
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
```
*(Leave `env` unset—o-series → dev, GPT-4* → prod.)*

---

## Progress & recovery

* Output flushes **row-by-row**—Ctrl-C and rerun skips completed pairs.  
* Live progress bar:
```
Processing 73%|██████████████▋ | 730/1000 [00:14<00:05, 49.2it/s]
```
o3-mini ≈ 100 ms/pair; embeddings ≈ 1 ms/string on M-series Mac.

---

## FAQ

| Q | A |
|---|---|
| **Why keep `similarity` in LLM mode?** | Free (~2 ms) sanity-check vs label. |
| **What causes `Unknown`?** | Empty/off-spec LLM reply—retry or switch model. |
| **Tune grey zone?** | Edit `hybrid_thresholds` or pass custom `--cfg`. |
| **Where are weights?** | Hugging Face cache (`~/.cache/huggingface`). |
| **Crash recovery?** | Output append-only; rerun resumes automatically. |

---

*Happy concording!  Stars ⭐, issues 🐞, and PRs 💡 welcome.*
# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

CONCORDIA is a lightweight engine that compares **any two functional-annotation sources**—old vs new, RAST vs UniProt, manual vs automated, etc.—and streams a tidy file with cosine-similarity scores **plus** one-word LLM labels.

---

## Installation

```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install        # install deps + CLI script
poetry shell          # activate the venv
```

---

## Quick-start recipes

| Purpose | Command |
|---------|---------|
| Compare a CSV with **default LLM** (o3-mini → apps-dev) | `concord example_data/annotations_full.csv` |
| Same CSV with **GPT-4o** (prod) | `concord …csv --llm-model gpt4o` |
| **Embed-only**, no LLM | `concord …csv --mode local` |
| **Hybrid** (embed first, LLM only for ambiguous pairs) | `concord …csv --mode hybrid` |
| Write to **custom file** | `concord …csv --output results.tsv` |
| **Ad-hoc strings** (LLM) | `concord --text-a "DNA repair protein RecA" --text-b "Recombinase A"` |
| Ad-hoc strings, **hybrid** | `concord --text-a "DNA ligase" --text-b "NAD-dependent ligase" --mode hybrid` |
| TSV input with explicit delimiter | `concord annotations_full.tsv --sep "\t"` |
| JSON list-of-objects | `concord annotations_full.json` |

---

## Accepted input formats

| Extension | Loader behaviour | Note |
|-----------|------------------|------|
| **`.csv`** | `pandas.read_csv(sep=",")` | default |
| **`.tsv` / `.tab`** | `pandas.read_csv(sep="\t")` | or `--sep "\t"` |
| **`.json`** | `pandas.read_json()` | list-of-objects **or** column-orient |

The first two textual columns that **don’t** end with “id” are selected, unless you pass `--col-a / --col-b`.

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
| `--mode` | `llm` (default) | `local` | `hybrid` |
| `--llm-model` | Gateway model (`gpto3mini`, `gpt4o`, …). |
| `--output` | Destination path (file-mode only). |
| `--cfg` | Alternate YAML config (default `concord/config.yaml`). |
| `--col-a / --col-b` | Explicit annotation columns. |
| `--sep` | Custom delimiter for text files. |

---

## Output fields

| Column | Meaning |
|--------|---------|
| *all originals* | Copied unchanged. |
| `similarity` | Cosine similarity (null in pure LLM mode). |
| `label` | One-word relation (table below). |
| `note` | Short reason from LLM (blank in local mode). |

### Label definitions

| Label | Meaning |
|-------|---------|
| **Identical** | Same function; minor wording differences. |
| **Synonym** | Same meaning via alternative phrasing. |
| **Partial** | Overlap; one broader/narrower. |
| **New** | Unrelated functions. |
| **Unknown** | LLM reply empty / unparseable. |
| **Error** | Exception captured; see `note`. |

### Label logic

* **local** – heuristic on similarity  
  * > 0.90 → Identical | 0.60 – 0.90 → Partial | < 0.60 → New  
* **llm** – first token from Argo Gateway reply.  
* **hybrid** – local rule when similarity outside gray zone; otherwise LLM.

---

## Prompting: how it works & how to tweak

All LLM instructions live in **`concord/llm/prompts.py`**.

```python
LABEL_SET = ["Identical", "Synonym", "Partial", "New"]

_FEW_SHOT = '''\
A: ATP synthase subunit beta
B: ATP synthase β subunit
Identical — wording difference only
'''

def build_annotation_prompt(a: str, b: str) -> str:
    return (
        f"{_FEW_SHOT}\n\n"
        f"Classify the relationship between these annotations.\n"
        f"A: {a}\nB: {b}\n\n"
        "Respond on one line as: <Label> — <very short reason>.\n"
        f"Allowed labels: {', '.join(LABEL_SET)}."
    )
```

* **Experimenting** – edit `_FEW_SHOT` or the wording of the final
  instruction, save, and rerun the CLI.  
* **Add chain-of-thought** – create a new builder (e.g.
  `build_cot_prompt`) and call it from `llm_label`.  
* **Force JSON output** – change the last instruction to
  “Respond in JSON: { "label": <>, "reason": <> }” and update
  `llm_label`’s parser.

No other code needs to change—prompt iteration is isolated to one file.

---

## Config snapshot (`concord/config.yaml`)

```yaml
engine:
  mode: hybrid
  hybrid_thresholds: {lower: 0.60, upper: 0.85}

llm:
  model: gpto3mini
  stream: false
  user: ${ARGO_USER}

local:
  model_id: NeuML/pubmedbert-base-embeddings
```

*(Leave `env` unset—client auto-selects: o-series → dev, GPT-4* → prod.)*

---

## Progress & recovery

* Results stream to disk **row-by-row**; you can kill & resume without losing progress.  
* `tqdm` shows live status:

```
Processing 73%|██████████████▋ | 730/1000 [00:14<00:05, 49.2it/s]
```

o3-mini ≈ 100 ms per pair; embeddings ≈ 1 ms per string on M-series CPU.

---

## FAQ

| Q | A |
|---|---|
| **Why keep `similarity` in `llm` mode?** | Free (~2 ms) sanity-check vs label. |
| **What causes `Unknown`?** | Empty or off-spec LLM reply. Retry or switch model. |
| **How to tune thresholds?** | Edit `hybrid_thresholds` or pass `--cfg custom.yml`. |
| **Where are model weights?** | Cached in `~/.cache/huggingface` on first run. |
| **Skip processed pairs after crash?** | Yes—rerun and pipeline resumes where it left off. |

---

*Happy concording! Issues & PRs welcome.*
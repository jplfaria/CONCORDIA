# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

CONCORDIA is a lightweight engine that compares **any two functional-annotation sources**—old vs. new, RAST vs. UniProt, manual vs. automated, etc.—and exports a tidy CSV with similarity scores and one-word labels.

---

## Installation

```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install      # installs dependencies + CLI script
poetry shell        # activate the virtual-env
```

---

## Quick-start recipes

| Purpose | Command |
|---------|---------|
| Compare a CSV with **default LLM** (o3-mini → apps-dev) | `concord example_data/changes.subset.csv` |
| Same CSV but **GPT-4o** (prod) | `concord example_data/changes.subset.csv --llm-model gpt4o` |
| **Embed-only**, no LLM | `concord example_data/changes.subset.csv --mode local` |
| **Hybrid** (embed first; LLM only for ambiguous) | `concord example_data/changes.subset.csv --mode hybrid` |
| Write to **custom file** | `concord example_data/changes.subset.csv --output results.csv` |
| **Ad-hoc strings** (LLM default) | `concord --text-a "DNA repair protein RecA" --text-b "Recombinase A"` |
| Ad-hoc strings, **hybrid** | `concord --text-a "DNA ligase" --text-b "NAD-dependent ligase" --mode hybrid` |

---

## CLI options

| Flag | Description |
|------|-------------|
| **`csv`** | Path to CSV with columns `gene_id,old_annotation,new_annotation`. |
| `--text-a / --text-b` | Compare two standalone strings instead of a CSV file. |
| `--mode` | `llm` (default) | `local` | `hybrid` |
| `--llm-model` | Override gateway model (`gpto3mini`, `gpt4o`, `gpt4`, …). |
| `--output` | Destination CSV (file-mode only). |
| `--cfg` | Alternate YAML config (default `concord/config.yaml`). |

---

## Output fields

| Column | Meaning |
|--------|---------|
| `gene_id` | Copied from input CSV (blank for ad-hoc). |
| `similarity` | Cosine similarity of PubMedBERT embeddings (`null` in pure LLM mode). |
| `label` | One-word relation (see definitions below). |

### Label definitions

| Label | Meaning | Typical use-case |
|-------|---------|------------------|
| **Identical** | Descriptions convey **exactly** the same function; mostly wording differences (“ATP synthase subunit beta” vs “ATP synthase β subunit”). |
| **Synonym** | Same functional meaning with clear alternative phrasing or recognised synonym, but not identical strings (“RecA protein” vs “DNA recombinase A”). |
| **Partial** | Overlap in meaning but one description contains extra qualifiers or missing context (“ABC transporter” vs “ABC transporter, maltose specific”). |
| **New** | Functions are unrelated or mutually exclusive; represents a genuine change in annotation. |
| **Unknown** | LLM returned an empty string (rare). Investigate or retry. |

### Label logic

* **local** – heuristic on `similarity`  
  * > 0.90 → Identical  
  * 0.60–0.90 → Partial  
  * < 0.60 → New
* **llm** – first word from Argo Gateway response (o-series or GPT-4*).  
* **hybrid** – local rule when similarity outside gray zone; otherwise LLM.

---

## Config snapshot (`concord/config.yaml`)

```yaml
engine:
  mode: hybrid
  hybrid_thresholds:
    lower: 0.60
    upper: 0.85

llm:
  model: gpto3mini      # auto-routes to apps-dev
  stream: false
  user: ${ARGO_USER}    # export ARGO_USER=<your-login>

local:
  model_id: NeuML/pubmedbert-base-embeddings
```

*(Leave `env` unset—client auto-selects: o-series → dev, GPT-4* → prod.)*

---

## Progress & performance

During embedding phases (local/hybrid) you’ll see a **tqdm** bar:

```
Processing  73%|██████████████▋   | 730/1000 [00:14<00:05, 49.2it/s]
```

o3-mini calls take ~100–150 ms per pair on the lab network; GPT-4o is similar.

---

## FAQ

| Q | A |
|---|---|
| **Why keep `similarity` in `llm` mode?** | It’s almost free (~2 ms) and helps QA the LLM label (e.g., LLM says **New** but similarity > 0.9). |
| **What does `Unknown` mean?** | Gateway returned an empty string—retry or switch model. |
| **How do I tune thresholds?** | Edit `hybrid_thresholds` in `config.yaml` or pass a custom file via `--cfg`. |
| **Where are the model weights?** | Not in the repo—sentence-transformers downloads them on first run to `~/Library/Caches/huggingface` (macOS) or `~/.cache/huggingface`. |

---

*Happy concording! Issues & PRs welcome.*
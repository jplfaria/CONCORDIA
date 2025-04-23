# CONCORDIA  
*CONcordance of Curated & Original Raw Descriptions In Annotations*

CONCORDIA is a lightweight engine that measures semantic agreement between **any two functional-annotation sources**—old vs new, RAST vs UniProt, manual vs. automated, etc.—and outputs an easy-to-scan CSV with similarity scores and one-word labels.

---

## Installation

```bash
git clone https://github.com/you/concordia.git
cd concordia
poetry install           # installs deps + CLI script
poetry shell             # activate the virtual-env

Quick-start recipes
-------------------

| Purpose | Command |
| --- | --- |
| Compare a CSV with default LLM (**o3-mini** on apps-dev) | `concord example_data/changes.subset.csv` |
| Same CSV, but GPT-4o (prod) | `concord example_data/changes.subset.csv --llm-model gpt4o` |
| Embed-only, no LLM | `concord example_data/changes.subset.csv --mode local` |
| Hybrid (embed first; LLM only for ambiguous) | `concord example_data/changes.subset.csv --mode hybrid` |
| Write to custom file | `concord example_data/changes.subset.csv --output results.csv` |
| Ad-hoc strings (LLM default) | `concord --text-a "DNA repair protein RecA" --text-b "Recombinase A"` |
| Ad-hoc strings, hybrid | `concord --text-a "DNA ligase" --text-b "NAD-dependent ligase" --mode hybrid` |

* * * * *

CLI Options
-----------

| Flag | Description |
| --- | --- |
| **`csv`** (positional) | Path to CSV with columns `gene_id,old_annotation,new_annotation`. |
| `--text-a / --text-b` | Compare two standalone strings instead of a CSV file. |
| `--mode` | `llm` (default) | `local` | `hybrid` |
| `--llm-model` | Override gateway model (`gpto3mini`, `gpt4o`, `gpt4`, ...). |
| `--output` | Destination CSV (file mode only). |
| `--cfg` | Alternate YAML config (default `concord/config.yaml`). |

* * * * *

Output Fields
-------------

| Column | Meaning |
| --- | --- |
| `gene_id` | Copied from input CSV (blank for ad-hoc). |
| `similarity` | Cosine similarity of PubMedBERT embeddings (`null` in pure LLM mode). |
| `label` | **Identical** |

### How labels are produced

-   **local** -- heuristic on similarity

    -   > 0.90 → Identical

    -   0.60 -- 0.90 → Partial

    -   < 0.60 → New

-   **llm** -- first token of Argo Gateway response (o-series or GPT-4*).

-   **hybrid** -- local rule when similarity outside gray-zone; otherwise LLM.

-   **Unknown** -- LLM returned empty text.

* * * * *

Config snapshot
---------------

yaml

CopyEdit

`engine:
  mode: hybrid
  hybrid_thresholds: { lower: 0.60, upper: 0.85 }

llm:
  model: gpto3mini      # auto-routes to apps-dev
  stream: false
  user: ${ARGO_USER}    # export ARGO_USER=<your-login>

local:
  model_id: NeuML/pubmedbert-base-embeddings`

*Leave `env` unset to let the client choose: o-series → dev, GPT-4* → prod.*

* * * * *

Progress & performance
----------------------

When embeddings are used (local/hybrid) you'll see a tqdm bar:

bash

CopyEdit

`Processing  73%|██████████████▋   | 730/1000 [00:14<00:05, 49.2it/s]`

o3-mini calls ~100--150 ms/pair on lab network; GPT-4o is similar.

* * * * *

FAQ
---

-   **Why keep similarity in `llm` mode?**\
    It's nearly free (~2 ms) and helps QA the LLM label.

-   **I get `Unknown` labels**\
    The gateway returned an empty string---retry or switch models.

-   **Can I tune thresholds?**\
    Edit `hybrid_thresholds` in `config.yaml` or override via `--cfg`.


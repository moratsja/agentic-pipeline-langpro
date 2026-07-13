# Agentic Pipeline LangPro

Agentic LLM knowledge-base generation for the
[LangPro](https://github.com/kovvalsky/LangPro) natural-logic theorem prover.

Given a premise/hypothesis pair, LangPro first attempts a proof with its
default lexical knowledge (WordNet). When the proof fails, an LLM proposes a
small set of lexical relations (`isa_wn`, `disj`) as KB injections; if the
proof still fails, a **critic** LLM analyses the failure and either authorises
a refined retry or stops. This generate–prove–refine loop is the *agentic*
pipeline.

## Repository structure

```text
├── prompts/                       # The three pipeline prompt templates
│   ├── knowledge_generation.txt   #   initial KB generation
│   ├── failure_analysis.txt       #   critic / failure analysis
│   └── knowledge_refinement.txt   #   KB refinement after critic feedback
├── src/
│   ├── agentic_pipeline_langpro/  # The agentic pipeline (CLI, agent loop, critic, ...)
│   └── kbprojection/              # Vendored substrate: LangPro API, LLM client,
│                                  #   KB filtering, SNLI/SICK loaders (MIT, E. Cesari)
├── data/
│   └── snli_train_entailment_1k.jsonl   # 1,000 sampled SNLI entailment problems
```

Run outputs are written to `results/` (created on demand, gitignored).

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                 # or: uv sync
```

### LLM access (OpenRouter)

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=...
```

The `.env` file at the repo root is loaded automatically.

### LangPro

By default the pipeline calls the **remote LangPro API**
(`https://langpro.hum.uu.nl/langpro-api/prove/`) — no local install needed.
WordNet on/off is controlled by `--langpro-builtin`, not `.env`.

## Running the pipeline

```bash
# WordNet ON (default)
agentic-pipeline-langpro --limit 5 --model google/gemini-3.1-flash-lite

# WordNet OFF
agentic-pipeline-langpro --langpro-builtin off --limit 5
```

Defaults to `data/snli_train_entailment_1k.jsonl` and writes JSONL to `results/`.

| Flag | Default |
| --- | --- |
| `--input` | `data/snli_train_entailment_1k.jsonl` |
| `--output` | `results/agentic_<timestamp>.jsonl` |
| `--model` | `LLM` from `.env` |
| `--max-iterations` | `3` |
| `--concurrency` | `4` |
| `--limit` | all problems |
| `--langpro-builtin` | `on` (`off` disables LangPro's built-in WordNet) |

Prompt templates are in `prompts/`.

## License

MIT (see [LICENSE](LICENSE)). The vendored `src/kbprojection` package is MIT,
Copyright (c) 2025–2026 Ettore Cesari.

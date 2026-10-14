# Research & Decision Intelligence Agent

Agentic AI Final Project — Project 3.

| Doc | Purpose |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | What we build (architecture, agents, tools, phases) |
| [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md) | How we build it (slices, grounding checks, offline/replay, habits) |
| [`docs/REQUIREMENTS_MATRIX.md`](docs/REQUIREMENTS_MATRIX.md) | Every brief requirement → code → test → demo scenario |
| [`eval/METRICS.md`](eval/METRICS.md) | Metric definitions and test sets |

> Status: **Phases 2–3 (research pipeline + orchestration)** — code router, Planner, LLM Orchestrator, 3 parallel specialists, Evidence Analyst, Writer, computed confidence.

## Setup (Windows, cmd)

```bat
cd research-agent
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and fill in:
- `GEMINI_API_KEY` — free key from Google AI Studio
- `TAVILY_API_KEY` — free key from app.tavily.com

## Check that everything works

```bat
python -m pytest -q
python scripts\smoke_test.py
```

The smoke test sends the primary demo question to Gemini through the OpenAI Agents SDK and checks that:
1. it returns a valid `ResearchPlan` (structured output),
2. the guard classifies a normal question as `allow` and an injection attempt as `block`,
3. tool calling works (the answer must contain a code only the tool returns),
4. a trace file is written to `data/traces/`.

If the main model is overloaded (503), rate-limited (429) or times out, each call automatically moves to the next model in `LLM_FALLBACKS` (e.g. `groq:qwen/qwen3.8-27b,gemini:gemini-3.5-flash-lite`); the busy model is skipped for `LLM_FALLBACK_COOLDOWN_SECONDS`. Runs print which models answered and how many switches happened.

### Record once, replay anywhere

```bat
set LLM_MODE=record
python scripts\smoke_test.py
set LLM_MODE=replay
python scripts\smoke_test.py
```

The second run makes no API calls and returns identical results. Use replay for tests and as the demo safety net.

### Phase 1 check (all 11 tools, live LLM, offline corpus)

```bat
python scripts\phase1_check.py
python scripts\phase1_check.py --live-search
```

### Model speed check

```bat
python scripts\latency_check.py
python scripts\latency_check.py --models gemini-3.5-flash-lite gemini-3.1-flash-lite --groq
```

Times one call per setting (reasoning off / none / low). Pick the fastest working one for `.env`.

### Full research run (Phases 2–3)

```bat
python scripts\run_research.py
python scripts\run_research.py --question secondary
python scripts\run_research.py --question sensitive
python scripts\run_research.py --question risk
python scripts\run_research.py --question "Your own question?"
```

Draft reports are saved to `samples/<run_id>.md` (+ a `.json` with the plan, workers, timeline and tool calls).

## Layout so far

```
src/research_agent/
  config.py    settings (load_dotenv first), code-level limits
  llm.py       Gemini/Groq via OpenAI-compatible API, shared rate limiter, retries
  models.py    LLM output models + internal records
  context.py   AppContext passed to tools (run id, store, tool log, review flags)
  store/db.py  SQLite: runs, sources, evidence, reviews, memory
  tracing.py   SDK tracing → local JSONL (+ optional OpenAI dashboard)
  guardrails/  url_safety.py (SSRF/local file blocking), injection.py (detect, sanitize, wrap)
  tools/       corpus, web (search/validate/scrape), quotes, quality, evidence, analysis, review,
               registry.py (the 11 function tools)
  agents/      planner, orchestrator, researchers (factory: technical/market/risk), writer,
               tool_agents (claim extractor, evidence analyst, support judge)
  workflows/   router.py (code routing), parallel.py (timeouts, isolation, timeline),
               confidence.py (computed confidence), pipeline.py (the full chain)
data/corpus/   15 synthetic test pages (see its README)
scripts/       smoke_test.py, check_models.py, phase1_check.py, run_research.py
tests/
```

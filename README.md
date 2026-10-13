# Research & Decision Intelligence Agent

Agentic AI Final Project — Project 3.

| Doc | Purpose |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | What we build (architecture, agents, tools, phases) |
| [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md) | How we build it (slices, grounding checks, offline/replay, habits) |
| [`docs/REQUIREMENTS_MATRIX.md`](docs/REQUIREMENTS_MATRIX.md) | Every brief requirement → code → test → demo scenario |
| [`eval/METRICS.md`](eval/METRICS.md) | Metric definitions and test sets |

> Status: **Phase 1 (corpus + tools)** — Phase 0 foundations plus an offline corpus, URL/injection safety and all 11 domain tools.

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

If Gemini is unavailable or rate-limited, switch providers in `.env`: `LLM_PROVIDER=groq` + `GROQ_API_KEY`.

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
  agents/      tool_agents.py (claim extractor, evidence analyst, support judge)
data/corpus/   15 synthetic test pages (see its README)
scripts/       smoke_test.py, check_models.py, phase1_check.py
tests/
```

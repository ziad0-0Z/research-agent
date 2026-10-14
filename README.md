# Research & Decision Intelligence Agent

Agentic AI Final Project — Project 3.

| Doc | Purpose |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | What we build (architecture, agents, tools, phases) |
| [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md) | How we build it (slices, grounding checks, offline/replay, habits) |
| [`docs/REQUIREMENTS_MATRIX.md`](docs/REQUIREMENTS_MATRIX.md) | Every brief requirement → code → test → demo scenario |
| [`eval/METRICS.md`](eval/METRICS.md) | Metric definitions and test sets |

> Status: **Phase 5 (guardrails + human review)** — input guardrail (regex + Guard Agent), code router, Planner, LLM Orchestrator, 3 parallel specialists, Evidence Analyst, Writer, Critic loop (verdict decided in code), hybrid BM25 + embedding retrieval, computed confidence, output guardrail + report-bound approval token, impact gate, persisted human review queue with resume.

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

### Embeddings for hybrid retrieval (RAG bonus)

Install Ollama from ollama.com, then:

```bat
ollama pull bge-m3
ollama list
```

`EMBED_MODE=auto` tries Ollama `bge-m3` (local, no quota) → Gemini embeddings API → BM25 only.
Retrieval never fails for lack of embeddings; it reports which method ran (`hybrid` or `bm25`).

```bat
python eval\retrieval_eval.py
```

Prints recall@5 for BM25-only and hybrid over the corpus, with every miss listed.

### Full research run (Phases 2–4)

```bat
python scripts\run_research.py
python scripts\run_research.py --question secondary
python scripts\run_research.py --question sensitive
python scripts\run_research.py --question risk
python scripts\run_research.py --question "Your own question?"
python scripts\run_research.py --plant-fault
```

`--plant-fault` is demo scenario 6: an unsupported claim is added to the first draft, citing a real
quote that says the opposite. The Critic must flag it (REVISE) and the Writer must remove or fix it.

Reports are saved to `samples/<run_id>.md` (final report if released, otherwise the draft) + a `.json`
with the plan, workers, timeline, Critic rounds, guardrail results and tool calls.

### Guardrails and human review (Phase 5)

```bat
python scripts\run_research.py --question attack            :: demo 7: blocked by the input guardrail, no agent runs
python scripts\run_research.py --question offscope          :: blocked by the Guard Agent
python scripts\run_research.py --fault fake_citation        :: demo 9: output guardrail rejects a fake citation
python scripts\run_research.py --fault tampered_quote       :: demo 9b: output guardrail catches an altered quote
python scripts\run_research.py --question sensitive         :: demo 3 + 11: paused for human review
python scripts\review.py list
python scripts\review.py show RUN_ID
python scripts\review.py approve RUN_ID --note "checked"
python scripts\review.py changes RUN_ID --note "add the PCI DSS limitation"   :: revise → Critic → back to the queue
python scripts\review.py reject RUN_ID --note "not enough evidence"
python eval\guardrail_eval.py --live                        :: catch rate + false positives, regex + Guard Agent
```

Release path, all in code: output guardrail (schema, citations exist, quotes still match the page,
no not-supported evidence, no claimed actions a tool did not confirm, no unknown URLs, no injected text)
→ approval token bound to the report's hash → impact gate (sensitive domain, Critic FAIL, guard outage,
high impact, confidence < 0.6) → human review queue or `generate_report`. A paused run resumes from
the database in a new process; it passes the output guardrail again and gets a fresh token.

## Layout so far

```
src/research_agent/
  config.py    settings (load_dotenv first), code-level limits
  llm.py       Gemini/Groq via OpenAI-compatible API, shared rate limiter, retries
  models.py    LLM output models + internal records
  context.py   AppContext passed to tools (run id, store, tool log, review flags)
  store/db.py  SQLite: runs, sources, evidence, reviews, memory
  tracing.py   SDK tracing → local JSONL (+ optional OpenAI dashboard)
  guardrails/  url_safety.py (SSRF/local file blocking), injection.py (detect, sanitize, wrap),
               input.py (regex pre-filter + Guard verdict in code), sdk.py (SDK input_guardrail on the
               Planner), output.py (release checks)
  tools/       corpus, web (search/validate/scrape), quotes, quality, evidence, analysis, review,
               registry.py (the 11 function tools)
  agents/      guard, planner, orchestrator, researchers (factory: technical/market/risk), writer, critic,
               tool_agents (claim extractor, evidence analyst, support judge)
  rag/         embeddings.py (Ollama → Gemini → none, vector cache), hybrid.py (BM25 + dense, RRF)
  workflows/   router.py (code routing), parallel.py (timeouts, isolation, timeline),
               eval_loop.py (Writer ⇄ Critic), confidence.py (computed confidence), pipeline.py,
               release.py (output guardrail → token → impact gate), resume.py (review queue + resume)
data/corpus/   15 synthetic test pages (see its README)
data/golden/   retrieval.json (recall@5 test set), guardrails.json (attacks + normal questions)
eval/          METRICS.md, retrieval_eval.py, guardrail_eval.py
scripts/       smoke_test.py, check_models.py, phase1_check.py, run_research.py, review.py
tests/
```

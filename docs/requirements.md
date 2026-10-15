# Requirements coverage

Every mandatory item in the project brief → where it is implemented → the test that proves it → the demo scenario that shows it.

## Brief §1 / §6 — Minimum technical requirements

| # | Requirement (brief) | Implemented in | Proven by | Demo |
|---|---|---|---|---|
| R1 | Agents SDK: `Agent`, `Runner`, `function_tool`, model settings | `llm.py`, `agents/` | `scripts/smoke_test.py` | 1 |
| R2 | ≥ 3 agents, distinct responsibilities | `agents/`: Guard, Planner, Orchestrator, 3 Researchers, Evidence Analyst, Writer, Critic, Claim Extractor, Support Judge | `test_pipeline_end_to_end_offline` | 1, 5 |
| R3 | ≥ 5 domain tools; ≥ 2 multi-tool workflows | `tools/registry.py` (11 tools) | `tests/test_tools.py`, `scripts/tools_check.py` | 2 |
| R4 | ≥ 2 Pydantic structured outputs, validated, invalid output handled | `models.py` (incl. `CriticReview`; Critic output re-checked in code by `_clean_review`) | `tests/test_models.py`, `test_clean_review_drops_unknown_ids_and_caps_support` | 2 |
| R5 | Input guardrail | `guardrails/input.py` (regex pre-filter → Guard Agent → verdict in code; Guard outage → forced review), `guardrails/sdk.py` (SDK `input_guardrail` on the Planner, blocking), pipeline stage 0 | `tests/test_guardrails.py` (attack library from `data/golden/guardrails.json`, false positives, fail-safe), `test_attack_is_blocked_before_any_agent_runs`; `eval/guardrail_eval.py` | 7 (`--question attack`) |
| R6 | Output guardrail | `guardrails/output.py` (schema, structure, citations exist, quotes still match, support, unconfirmed actions, links, injection) in `workflows/release.py`; pass → approval token | `tests/test_guardrails.py` (one test per check), `test_output_guardrail_rejects_faults_planted_after_the_critic` | 9 (`--fault fake_citation` / `tampered_quote`) |
| R7 | ≥ 1 protected tool/action | `scrape_page` (URL safety), `save_evidence` (quote check), `generate_report` (token bound to report hash + human approval) | `tests/test_tools.py`, `tests/test_safety.py`, `test_token_is_void_if_report_changes_after_approval`, `test_paused_run_cannot_be_released_by_calling_generate_report` | 8 |
| R8 | `trace()` around major workflows, shown in demo | `tracing.py` (trace_start records run id as group_id), `workflows/pipeline.py` + `resume.py` (trace ids saved in run state), `trace_view.py` (find a run's traces: linked / group / time; span tree, tokens, errors), Traces page in the UI | smoke test; `tests/test_reliability.py` (`test_trace_summary_reads_tokens_tools_errors_and_names_tasks`, `test_traces_found_by_link_group_or_time`) | 12 |
| R9 | Context (`RunContextWrapper`) | `context.py`, tools receive `RunContextWrapper[AppContext]` | `test_generate_report_tool_wrapper_returns_error_not_crash` | 2 |
| R10 | Session memory (`SQLiteSession`) | `workflows/session.py`: Planner runs with `session=` on follow-ups; one assistant turn per run (released report summary, or status only); blocked questions never recorded; Guard sees the previous question as context; `--follow-up latest` / `--session` in `run_research.py`; resume reports the outcome to the conversation | `tests/test_session.py` (incl. `test_real_sdk_planner_run_with_session_gets_history_and_passes_the_guardrail`, `test_blocked_question_never_enters_the_conversation`, `test_without_a_session_prompts_are_unchanged`) | follow-up |
| R11 | Prompt chaining (2+ dependent steps, intermediate output shown) | `workflows/pipeline.py`: plan → evidence → analysis → report | `test_pipeline_end_to_end_offline` (stage order) | 2 |
| R12 | Routing (explained) | `workflows/router.py` (code: sensitive domains, ambiguous terms need a context word) + Planner types (LLM) | `test_router_*`, `test_code_router_meets_target_on_golden_set_including_keyword_traps`, `test_tightened_router_still_flags_real_sensitive_wording`; routing accuracy in `eval/run_eval.py` (code router 20/20 offline, 4/4 recorded, 19/20 live) | 3 |
| R13 | Parallelization (`asyncio.gather`) | `workflows/parallel.py` | `test_parallel_isolates_failures_and_timeouts` | 4 |
| R14 | Orchestrator-workers (2+ distinct workers) | `agents/orchestrator.py` + `agents/researchers.py` | `test_validate_assignments_fixes_llm_mistakes` | 5 |
| R15 | Evaluator-optimizer with stop condition + max iterations | `workflows/eval_loop.py`: Writer ⇄ Critic, PASS/REVISE/FAIL decided in code; stops on PASS, max rounds, no progress, Writer error | `test_planted_unsupported_claim_is_caught_and_fixed`, `test_loop_stops_at_max_rounds_and_keeps_best_draft`, `test_loop_survives_critic_error_and_writer_error` | 6 (`--plant-fault`) |
| R16 | ≥ 1 code-orchestrated decision | `route_question`, `validate_assignments`, confidence, Critic verdict + strength rules (`eval_loop.py`), input verdict, output guardrail, impact gate (`release.py`) | `test_router_flags_sensitive_domains`, `test_partial_support_is_never_strong`, `test_high_impact_report_needs_review`, `test_low_confidence_needs_review` | 3, 6, 11 |
| R17 | ≥ 1 LLM-orchestrated decision | Planner types, Orchestrator assignments + depth | `test_pipeline_end_to_end_offline` | 5 |
| R18 | Human-in-the-loop stop | `workflows/release.py` (impact gate → `request_human_approval`), `workflows/resume.py` (approve / reject / request changes, resume from the DB), `scripts/review.py` | `tests/test_review.py` (`test_approve_then_resume_in_a_fresh_process`, `test_changes_requested_revises_and_returns_to_the_queue`, `test_reject_releases_nothing`, `test_review_rules`) | 11 |
| R19 | Reliability: tool failure, malformed input, missing data, model error, empty/invalid output | `llm.py` (model fallback chain + cooldown), `workflows/parallel.py` (worker isolation), Critic loop (invalid citations → REVISE; Critic/Writer errors keep best draft), `rag/embeddings.py` (embedding fallback chain → BM25), `workflows/reliability.py` (per-run record: workers, search providers, fallbacks, model/embedding failures; report gets a "Research coverage" section when degraded), `--fault search_outage` (Tavily + DuckDuckGo down → offline corpus; risk specialist loses every provider) | `test_overloaded_main_model_falls_back_and_cools_down`, `test_pipeline_survives_a_failed_worker`, `test_critic_loop_fixes_invented_citation`, `test_chain_falls_back_to_next_provider_then_to_bm25`, `test_killing_search_still_yields_a_labelled_partial_report` | 10 (`--fault search_outage`) |
| R20 | Usable UI (Gradio) | `app.py`, `ui/app.py` (sidebar + 10 pages; report tab Evidence graph; Evaluation page with targets chart + 12-scenario table linked to runs), `ui/evidence_graph.py`, `ui/data.py` (read models from the stores), `ui/jobs.py` (background runs fed by the pipeline's real events), `ui/components.py`, `ui/trace_html.py`, `ui/styles.css` | `tests/test_ui.py` (empty store shows no invented numbers, read models follow a real offline run, untrusted text escaped, rail states, app builds); UI driven end-to-end with the offline fake model (run → paused → approve → released) | all (scenario dropdown on New research; scenario 8 runs the protected tools live) |

## Brief §5 — Project 3 specifics

| # | Requirement | Implemented in | Proven by | Demo |
|---|---|---|---|---|
| P1 | Research plan, source collection, evidence summaries, final report | `workflows/pipeline.py` | `test_pipeline_end_to_end_offline` | 1 |
| P2 | Tools: search_web, scrape_page, extract_claims, compare_sources, save_evidence, generate_report + validation tool | `tools/registry.py` (+ validate_source, validate_citation) | `tests/test_tools.py` | 2 |
| P3 | Chain: question → subquestions → evidence → synthesis → report | `workflows/pipeline.py` | `test_pipeline_end_to_end_offline` | 2 |
| P4 | Route subquestions to specialists | Planner types + `validate_assignments` | `test_validate_assignments_fixes_llm_mistakes` | 3 |
| P5 | Subquestions researched concurrently | `workflows/parallel.py` | `test_parallel_isolates_failures_and_timeouts` | 4 |
| P6 | Orchestrator delegates to specialist research agents | `agents/orchestrator.py` | `test_pipeline_end_to_end_offline` | 5 |
| P7 | Critic: unsupported claims, missing sources, contradictions; bounded revisions | `agents/critic.py` + `workflows/eval_loop.py` (code checks + RAG retrieval + LLM review) | `test_planted_unsupported_claim_is_caught_and_fixed`, `test_code_checks_flag_ignored_contradiction`, `test_code_checks_flag_invented_citation` | 6 |
| P8 | Guardrails: injection in web pages, invented citations, unsupported claims, report structure | `injection.py` on scraped pages, input guardrail, Critic loop, output guardrail (citations, quotes, support, structure, injection leak) | `test_scrape_injected_page_is_flagged_and_sanitized`, `tests/test_guardrails.py` | 7, 8, 9 |
| P9 | Human approval before high-impact recommendation | impact gate in `release.py` (sensitive · Critic FAIL · guard outage · high impact · confidence < 0.6); `generate_report` re-checks approval | `test_high_impact_report_needs_review`, `test_sensitive_run_pauses_and_appears_in_the_queue` | 11 |

## Brief §8 — Reliability & safety rules

| # | Rule | Where enforced | Proven by |
|---|---|---|---|
| S1 | Never claim an action happened unless a tool confirmed it | tool log in `AppContext`; output guardrail `unconfirmed_actions` (approval claims need an approved review; "we tested/benchmarked…" always blocked); Writer told it only reads sources | `test_claims_of_unconfirmed_actions_rejected`, `test_human_approval_claim_allowed_once_approved` |
| S2 | Never invent data / citations | `save_evidence` quote check, `check_citation` | `test_save_evidence_rejects_fabricated_quote`, `test_validate_citation_unknown_id_skips_llm` |
| S3 | LLM can't bypass deterministic rules | regex before the Guard; Guard verdict interpreted in code; router, Critic verdict, output guardrail, token, impact gate all in Python; token never persisted (resume re-checks) | `test_attacks_blocked_by_regex_without_calling_the_guard`, `test_paused_run_cannot_be_released_by_calling_generate_report` |
| S4 | Guardrails meaningful and tested | `tests/test_guardrails.py`, `tests/test_review.py`, `eval/guardrail_eval.py` | 52 + 16 tests; regex catch 11/15 with 0/15 false positives (rest is the Guard's job, measured live) |
| S5 | Tool arguments validated before sensitive actions | strict tool schemas + checks in each tool | `test_eleven_tools_with_strict_schemas`, `test_scrape_blocks_unsafe_urls` |
| S6 | Every evaluator loop has max iterations | `MAX_CRITIC_ROUNDS` (default 3) + hard cap `MAX_ROUNDS_CAP = 5` in `eval_loop.py`; citation repair folded into the same loop | `test_loop_stops_at_max_rounds_and_keeps_best_draft`, `test_loop_respects_hard_round_cap` |
| S7 | Graceful recovery after tool/model failure | search fallback chain, model fallback chain (503/429/timeout → next model), worker isolation, tool errors returned not raised | `test_live_mode_falls_back_to_offline`, `test_pipeline_survives_a_failed_worker` |
| S8 | Prompt injection in user input, tool results, retrieved content | input guardrail (regex + Guard, question wrapped as data) + `guardrails/injection.py` on scraped pages + output guardrail injection-leak check | `tests/test_safety.py`, `test_scrape_injected_page_is_flagged_and_sanitized`, `test_injected_text_in_report_rejected` |
| S9 | No API keys or real customer data in repo | `.gitignore`, `.env.example` | |

## Brief §10 — Bonus

| Bonus | Implemented in | Proven by |
|---|---|---|
| RAG / evidence retrieval (the chosen bonus) | `rag/` (Ollama bge-m3 → Gemini → BM25, RRF fusion, quality weight) used by the Critic per finding | `tests/test_rag.py`; recall@5 in `eval/retrieval_eval.py` (BM25 91%, hybrid 91%) |
| Long-term memory (architecture layer, not claimed as the bonus) | `memory/`: policy (completed runs only; verified, not not_supported, no injection), TTL by claim type (pricing 30 d · market/risk 90 d · technical 180 d), lookup after planning → `save_evidence` (re-verified) → normal Critic + output guardrail, stale items re-fetched and re-verified in code, memory-covered subquestions researched at quick depth, supersede decided in code (contested otherwise), delete with tombstone; `scripts/memory.py` | `tests/test_memory.py` (48 tests, incl. exit criterion `test_second_similar_question_reuses_fresh_evidence`); `eval/memory_eval.py` (false reuse 0/8, precision 0.74, recall 0.7; related reports P 1.0 R 0.9) |


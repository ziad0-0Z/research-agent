# Design decisions

Each entry: what was decided, the alternatives, and why. Where a number drove the choice, it is given.

## Platform

**D1. Free model stack: Gemini through its OpenAI-compatible endpoint, Groq as fallback.**
The OpenAI API is paid. The Agents SDK accepts any OpenAI-compatible endpoint through
`OpenAIChatCompletionsModel`, so Gemini's free tier works without changing agent code. Free models are often
overloaded (503) or rate-limited (429), so no single model is trusted: every call can fall back to the next one in
`LLM_FALLBACKS`. Groq rejects tools and a JSON output schema in the same request, so it is skipped for those calls only.

**D2. One shared rate limiter for every agent.**
Parallel specialists on a free tier hit 429 errors almost immediately without it. A single limiter (requests per
minute + concurrency cap) is simpler than per-agent budgets and makes the cost of parallelism visible.

**D3. Search chain Tavily → DuckDuckGo → offline corpus.**
Live search gives real research; the offline corpus makes tests deterministic and guarantees the demo. The report says
which provider answered.

**D4. Record and replay model responses.**
Replay keys are a hash of the exact request, so prompts never contain run ids or timestamps. A replay miss raises an
error and never silently calls the API.

## Control: what the model decides and what code decides

**D5. "The model decides what to do; Python decides what is allowed."**

| Decision | Made by | Reason |
|---|---|---|
| Sensitive domain → human review | Python | safety rule, must be auditable |
| URL allowed to be fetched | Python | security (SSRF, local files) |
| Quote exists in the page | Python | grounding |
| Citations valid, schema valid | Python | hard rule |
| Confidence score | Python | models rate themselves poorly |
| Human review needed | Python | safety rule |
| What enters long-term memory | Python | memory policy |
| Round limits, timeouts | Python | bounded execution |
| Split question into subquestions | model | interpretation |
| Subquestion → specialist | model | semantic classification |
| Which specialists, how deep | model | flexible planning |
| Interpreting contradictions | model | semantic judgement |

**D6. Regex before the Guard Agent.**
A regex hit is free, deterministic and cannot be argued with; the Guard handles meaning (off-scope, harmful, a
conclusion requested in advance without trigger words). Fabrication patterns need an action verb plus a citation noun,
so "detect fake reviews" and "use synthetic test data" pass. Measured: regex alone catches 11/15 attacks with 0/15
false positives; with the Guard, 15/15 and 0/15.

**D7. Guard outage is fail-safe, not fail-open or fail-closed.**
If the Guard Agent errors after the regex passed, the run continues but a human must approve the result. Blocking every
user during an outage (fail-closed) or skipping the check silently (fail-open) were both worse.

**D8. The Critic judges, code decides the verdict.**
The model gives a support level per citation; PASS / REVISE / FAIL is computed from blocking issues. The model cannot
wave a report through. The contradiction check blocks only when the report lists no contradictions at all, so a
spurious Analyst contradiction cannot force a FAIL.

**D9. Confidence is computed, and says what it means.**
Coverage × source quality × contradiction penalty × worker factor × corroboration. It describes how well the report is
backed by what it cites, not the probability the recommendation is correct; it is not calibrated against outcomes.

**D10. Below 50 % coverage the report may not pick an option.**
Code replaces the recommendation with "Insufficient evidence to choose" and lists what is missing.

## Safety

**D11. The output guardrail is a gate, not a fixer.**
Repairs belong in the Critic loop. Rejection by the output guardrail is final for that run, which keeps the last check
simple to reason about.

**D12. The approval token is bound to the report's hash and never stored.**
If anything changes the report after approval, the token is void. A resumed run must pass every check again.

**D13. A report revised after a human asked for changes always goes back to a human.**

**D14. URL safety blocks every non-public address in every notation.**
Including `127.1`, `0177.0.0.1`, `0x7f.0.0.1`, `2130706433`, carrier-grade NAT and IPv4-mapped IPv6 metadata addresses,
also when DNS is not resolved. Known remaining gap: DNS rebinding between the check and the HTTP client's own lookup.

**D15. Scraped text is untrusted data.**
Injection patterns are detected, the text is wrapped in `<untrusted_content>`, and the source's quality score is
lowered. The output guardrail also checks that no injected instruction reached the report.

## Retrieval and memory

**D16. RAG is the one bonus claimed; memory and advanced parallelism are architecture.**
The brief asks for one bonus. Hybrid retrieval (BM25 for numbers and exact terms, embeddings for paraphrases, fused
with reciprocal rank fusion, weighted 0.8 relevance + 0.2 source quality) is used by the Critic. Recall@5: 91 % for both
BM25 and hybrid on 22 relevant pages; hybrid fixed one miss and introduced another.

**D17. Memory stores evidence, not conclusions.**
A past report given to the Writer could be copied without citations. Past reports are only shown to the user.

**D18. Reused evidence goes through `save_evidence`, not a side door.**
The Critic, the output guardrail and confidence then treat it like new evidence, with no special cases.

**D19. Memory-covered subquestions are still researched, at quick depth.**
Skipping them saves more time, but a fresh look catches changes that can supersede what was remembered.

**D20. IDF-weighted word gate for memory reuse.**
A plain shared-word ratio at 0.34 reused memory for 3 of 8 unrelated subquestions (words like "cost", "million").
IDF weighting at 0.4 gave 0 of 8.

**D21. Supersede only when there is a clear winner.**
Newer evidence replaces older only when the Analyst paired them as a contradiction, the newer one is supported, and it
is from the same site or (for pricing) from an equal-or-better source. Otherwise both are kept as contested.

## Evaluation

**D22. Every number says how it was measured.**
Recorded (from stored runs), offline (golden set + code) or live (model calls). Model errors during live evaluation are
counted separately and never scored as passes.

**D23. The golden set found a real bug.**
The keyword router sent "Phi-3", "tennis court", "trading card" and "diagnose latency" to the sensitive path (16/20).
Ambiguous keywords now need a context word (20/20). Caveat: the traps were written knowing the keyword list.

## Interface

**D24. One background event loop for the UI.**
The shared rate-limited model client must live in one asyncio loop; the UI runs one job at a time and polls the
pipeline's real events every second instead of simulating progress.

**D25. Square nodes for Python, round nodes for the model.**
The pipeline rail in the UI uses the same distinction as these docs, so a viewer can see who decided each step.

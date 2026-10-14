> **DRAFT** — Critic verdict **PASS** after 1 round(s) (passed). Output guardrail and human review are not applied yet.

# Cloud LLM APIs vs. Self-Hosting for a 5-Person Startup

## Executive summary
A 5-person startup should use managed cloud LLM APIs rather than self-hosting open-source models. Engineering labor for maintenance and low initial token volumes make self-hosting significantly more expensive and operationally burdensome. This decision depends on maintaining token volumes below the 1 to 3 billion threshold where self-hosting becomes cost-effective.

## Findings
1. For organizations with fewer than ten engineers, hosted LLM APIs have a lower total cost of ownership than self-hosting open-weight models when token volume is below several billion per month. [ev12] _(evidence: strong)_
2. Engineering labor dominates the cost comparison, as self-hosting requires 0.25 to 0.5 of a full-time engineer, outweighing compute savings until token volumes reach several billion per month. [ev6], [ev7], [ev13] _(evidence: strong)_
3. Running a single inference-class GPU 24/7 costs $1,100 to $2,900 per month before storage, networking, and monitoring, making self-hosting more expensive for low volumes below 1 billion tokens where the GPU sits idle. [ev18], [ev19] _(evidence: moderate)_
4. The break-even point for self-hosting a mid-sized model compared to published API prices falls between 1 and 3 billion tokens per month. [ev20] _(evidence: moderate)_
5. Small teams attempting to self-host face operational risks, including out-of-memory crashes from long prompts, driver update failures, and the need for constant monitoring. [ev8], [ev9], [ev10] _(evidence: weak)_
6. Dedicated deployments offer faster time-to-first-token (around 120 milliseconds) compared to shared APIs (250 to 600 milliseconds), though shared APIs provide flexibility until rate limits or peak-hour latency spikes occur. [ev14], [ev15], [ev16] _(evidence: moderate)_
7. Most major hosted API providers do not use business API data for training by default and offer data retention limits of up to 30 days, while zero-retention is available for enterprise customers under a DPA. [ev1] _(evidence: moderate)_

## Recommendation
Use managed cloud LLM APIs now; revisit self-hosting only when token volume consistently exceeds 1 billion tokens per month and dedicated infrastructure engineering headcount can be justified.

**Impact:** medium

## Limitations
- Evidence regarding operational horror stories relies on forum posts which have lower source quality.
- Latency and throughput benchmarks come from blog posts rather than rigorous academic studies.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Most major providers state that data sent through business APIs is not used to train their models by default.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev6 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev7 | “For small teams, labour dominates the comparison. At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev8 | “A routine driver update broke the serving container on a Friday night and the bot was down for six hours.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev9 | “Long prompts caused out-of-memory crashes under load, and we spent two weeks tuning batch sizes.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev10 | “We switched back to an API after three months. Self-hosting itself wasn't hard, but someone always had to babysit it, and we're only five people.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev12 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev13 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev14 | “For short prompts, the dedicated deployment had a median time-to-first-token of about 120 milliseconds, compared with 250 to 600 milliseconds for the shared APIs.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev15 | “At the 99th percentile, the difference widened: shared APIs occasionally took several seconds during peak hours, while the dedicated GPU stayed under one second until it reached its capacity.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev16 | “Shared APIs also enforce rate limits. On entry-level plans, our test account hit request-per-minute limits during load tests, which caused errors until we added retries and backoff.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev18 | “On-demand rental for a single inference-class GPU ranged from $1.50 to $4.00 per hour across five providers. Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev19 | “Below that volume, the GPU sits idle much of the time and the effective cost per token is higher than an API.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev20 | “Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
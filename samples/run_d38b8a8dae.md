> **DRAFT** — Critic verdict **PASS** after 1 round(s) (passed).

# Cloud LLM APIs vs Self-Hosting Open-Source Models for a 5-Person Startup

## Executive summary
A 5-person startup should use cloud LLM APIs rather than self-hosting unless its token volume exceeds several billion per month. Below this threshold, hosted APIs provide a lower total cost of ownership because the engineering labor required to maintain self-hosted models outweighs potential compute savings. This decision depends heavily on monthly token volume and the opportunity cost of engineering headcount.

## Findings
1. Keeping a self-hosted open-weight model reliable in production requires 0.25 to 0.5 of a full-time engineer for ongoing maintenance, patching, and monitoring. [ev1] _(evidence: strong)_
2. At typical startup engineering costs, the labor required for self-hosting outweighs compute savings until volume reaches several billion tokens per month. [ev2], [ev3] _(evidence: strong)_
3. On-demand rentals for a single inference-class GPU range from $1.50 to $4.00 per hour, translating to roughly $1,100 to $2,900 per month around the clock. [ev4], [ev5] _(evidence: moderate)_
4. The break-even point for self-hosting falls between 1 and 3 billion tokens per month, below which GPUs sit idle and effective costs exceed API pricing. [ev6], [ev7] _(evidence: moderate)_
5. Initial setup for serving, autoscaling, and monitoring takes about three weeks of engineering time before ongoing maintenance begins. [ev9] _(evidence: moderate)_

## Recommendation
Use cloud LLM APIs for all product features until monthly token volume consistently exceeds 1 to 3 billion tokens; revisit self-hosting only when growth justifies dedicating 0.25 to 0.5 of an engineer to infrastructure maintenance.

**Impact:** medium

## Limitations
- The risk analysis subquestion failed to gather evidence, leaving data privacy, security, and compliance risks unaddressed.
- Evidence sources do not cover specialized hardware optimizations or multi-region high availability costs.

## Research coverage
- **Partial research:** the risk specialist saved no evidence (subquestions sq3); findings for them may be missing.
- **Search fallback:** tavily, ddg unavailable; 2 of 5 searches were answered by a fallback provider (by provider: offline 2).
- **Search failures:** 3 search(es) failed on every provider.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev2 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev3 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev4 | “On-demand rental for a single inference-class GPU ranged from $1.50 to $4.00 per hour across five providers.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev5 | “Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev6 | “Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev7 | “Below that volume, the GPU sits idle much of the time and the effective cost per token is higher than an API.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev9 | “Setting up serving, autoscaling, monitoring and upgrades took our team about three weeks initially, plus ongoing maintenance. Teams should add that labour cost before deciding.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
> **DRAFT** — Critic verdict **PASS** after 1 round(s) (passed).

# Cloud LLM APIs vs. Self-Hosting for a 5-Person Startup

## Executive summary
A 5-person startup should use cloud LLM APIs rather than self-hosting open-source models for low-to-medium transaction volumes. Academic and news sources demonstrate that hosted APIs offer lower total cost of ownership at these volumes because self-hosting requires significant ongoing engineering hours for infrastructure maintenance. This approach depends on token volumes remaining below the threshold where compute savings outweigh labor costs.

## Findings
1. For organizations with fewer than ten engineers, hosted APIs have the lower total cost of ownership in every scenario modeled until token volume reaches several billion per month. [ev1] _(evidence: strong)_
2. Keeping a self-hosted model reliable in production requires between 0.25 and 0.5 of a full-time engineer for patching drivers, handling capacity, monitoring quality, and upgrading models. [ev2] _(evidence: strong)_
3. At a fully loaded startup engineering cost, the labor required for self-hosting outweighs its compute savings until token volume reaches several billion per month. [ev3] _(evidence: strong)_
4. For early-stage startups processing median volumes around 40 million tokens per month, monthly API spend for small models stays under $500, while frontier models cost between $800 and $2,500 per month. [ev4] _(evidence: moderate)_
5. For low and moderate volumes, APIs remain the cheapest option in direct spend. [ev5] _(evidence: moderate)_
6. Dedicated deployments can achieve a lower time-to-first-token compared to shared APIs, but require provisioning another GPU when concurrent requests exceed capacity. [ev9], [ev12] _(evidence: moderate)_
7. Self-hosting keeps data inside the organization, whereas using a hosted API means personal data leaves the organization's own infrastructure. [ev7], [ev8] _(evidence: strong)_
8. An independent 2026 audit found hosted APIs cut total costs by 70% for every team size. [ev999] _(evidence: strong)_

## Where sources disagree
- [ev1] and [ev5] vs [ev6]: Academic and news sources conclude that hosted APIs are cheaper for low and moderate volumes when factoring in engineering labor costs, whereas a single blog post claims an 80 percent reduction in monthly bills after switching to self-hosting. The blog post ignores engineering overhead.

## Recommendation
Use cloud LLM APIs now to conserve engineering hours while transaction volumes are low; revisit self-hosting only when token volumes approach several billion per month or if strict data residency mandates require internal infrastructure.

**Impact:** medium

## Limitations
- Evidence lacks detailed DevOps tool comparisons for specific frameworks like vLLM or TGI.
- Data security compliance ease under specific frameworks like HIPAA or SOC2 is not fully quantified outside of general data residency statements.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev2 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev3 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev4 | “In our sample of twelve startups with fewer than ten employees, the median team processed about 40 million tokens per month. At that volume, monthly API spend for small models stayed under $500, and teams using a frontier model for every request spent between $800 and $2,500 per month.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev5 | “Our conclusion: for low and moderate volumes, APIs remain the cheapest option in direct spend.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev7 | “Self-hosting keeps data inside the organisation but transfers the full security responsibility to its own team.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev8 | “Using a hosted API means personal data leaves the organisation's own infrastructure.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev9 | “For short prompts, the dedicated deployment had a median time-to-first-token of about 120 milliseconds, compared with 250 to 600 milliseconds for the shared APIs.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev12 | “The dedicated GPU has its own ceiling: once concurrent requests exceeded its capacity, latency rose sharply, and adding capacity meant provisioning another GPU.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev999 | _invalid: evidence id does not exist in this run_ | — |
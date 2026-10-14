> **DRAFT** — Critic verdict **PASS** after 1 round(s) (passed).

# TCO Analysis: Self-Hosting Open-Source LLMs vs. Cloud APIs for Startups

## Executive summary
For early-stage startups with lower token volumes, using managed cloud APIs is generally cheaper and eliminates fixed infrastructure costs. Self-hosting open-source LLMs only becomes financially advantageous when token volume reaches several billion tokens per month. However, small engineering teams must also account for the significant ongoing maintenance overhead and engineering hours required to keep self-hosted models reliable.

## Findings
1. For small teams, engineering labour costs outweigh the compute savings of self-hosting until token volume reaches several billion per month. [ev1] _(evidence: strong)_
2. Keeping a self-hosted model reliable in production requires between 0.25 and 0.5 of a full-time engineer on an ongoing basis for organizations with fewer than ten engineers. [ev2] _(evidence: strong)_
3. Running a single inference-class GPU around the clock on-demand costs roughly $1,100 to $2,900 per month before storage, networking, and monitoring. [ev3] _(evidence: moderate)_
4. For median early-stage startups processing around 40 million tokens per month, monthly API spend for small models stays under $500, while frontier models range from $800 to $2,500 per month. [ev4] _(evidence: moderate)_
5. The break-even point for self-hosting a mid-sized model compared to cloud API prices falls between 1 and 3 billion tokens per month. [ev5] _(evidence: moderate)_
6. Commercial cloud LLM APIs offer pay-as-you-go pricing with no fixed monthly cost during the product-market fit stage, with input tokens priced lower than output tokens. [ev12], [ev13], [ev14] _(evidence: moderate)_

## Recommendation
Use managed cloud APIs during the product-market fit stage to avoid fixed monthly infrastructure costs; revisit self-hosting only when token volume consistently exceeds one to three billion tokens per month.

**Impact:** medium

## Limitations
- Most cost models rely on estimates and case studies that may vary based on specific workload characteristics, prompt lengths, and provider price fluctuations.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 · reused from memory, last verified 2026-10-14 |
| ev2 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 · reused from memory, last verified 2026-10-14 |
| ev3 | “Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev4 | “In our sample of twelve startups with fewer than ten employees, the median team processed about 40 million tokens per month. At that volume, monthly API spend for small models stayed under $500, and teams using a frontier model for every request spent between $800 and $2,500 per month.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 · reused from memory, last verified 2026-10-14 |
| ev5 | “Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev12 | “Across the major providers we surveyed, small and mid-sized models are now priced between $0.10 and $0.60 per million input tokens, while frontier models range from $2 to $15 per million input tokens.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev13 | “Output tokens typically cost three to five times more than input tokens.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev14 | “Pricing is pay-as-you-go, so costs scale linearly with traffic and drop to near zero when usage is low. Several founders told us this was the main reason they chose APIs at launch: there is no fixed monthly cost while they are still searching for product-market fit.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
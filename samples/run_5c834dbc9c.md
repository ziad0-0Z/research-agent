> **Confidence 0.73** (computed in code): 3/3 subquestions covered by usable cited evidence; mean source quality 0.73; 0 contradiction(s); 0 failed/timed-out worker(s).

# Cost Breakdown and Volume Thresholds for Self-Hosting Open-Source LLMs vs. Cloud APIs

## Executive summary
For early-stage startups processing under 1 to 3 billion tokens per month, pay-as-you-go cloud APIs offer a lower total cost of ownership than self-hosting open-source LLMs. While on-demand GPU rentals cost between $1,100 and $2,900 per month per GPU, engineering maintenance—estimated at 0.25 to 0.5 of a full-time employee—dominates early-stage costs. Therefore, startups should rely on cloud APIs during product-market fit searches and only consider self-hosting when monthly token volume reliably exceeds several billion tokens.

## Findings
1. Running a single inference-class GPU on-demand costs roughly $1,100 to $2,900 per month, with rental prices ranging from $1.50 to $4.00 per hour. [ev2], [ev4] _(evidence: moderate)_
2. A single high-memory GPU can serve a 7-14B parameter model, whereas 70B-class models require two to four GPUs, multiplying the infrastructure bill. [ev3] _(evidence: moderate)_
3. Cloud APIs feature no fixed monthly costs, with median early-stage startups processing 40 million tokens per month spending under $500 monthly for small models. [ev5], [ev8] _(evidence: moderate)_
4. Small and mid-sized hosted models cost between $0.10 and $0.60 per million input tokens, with output tokens costing three to five times more. [ev12], [ev13] _(evidence: moderate)_
5. The break-even point where self-hosting becomes cheaper than cloud APIs falls between 1 and 3 billion tokens per month. [ev1], [ev6], [ev7] _(evidence: strong)_
6. Initial setup of serving, autoscaling, monitoring, and upgrades takes about three weeks of engineering time. [ev11], [ev16] _(evidence: moderate)_
7. Maintaining a self-hosted model in production requires 0.25 to 0.5 of a full-time engineer, whose labor costs outweigh compute savings below several billion tokens per month. [ev9], [ev10], [ev15] _(evidence: strong)_

## Recommendation
Use pay-as-you-go cloud APIs while searching for product-market fit and maintaining low token volumes. Revisit self-hosting and evaluate open-source LLMs only when monthly token volume consistently exceeds 1 billion tokens and engineering capacity allows for dedicated maintenance overhead.

**Impact:** medium

## Limitations
- Evidence relies on general market rates and academic models from early 2026, which may fluctuate based on specific cloud provider pricing updates.
- Data does not cover enterprise-scale workloads exceeding tens of billions of tokens.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Below that volume, the GPU sits idle much of the time and the effective cost per token is higher than an API.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev2 | “Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev3 | “A single high-memory GPU comfortably serves a 7–14B parameter model, while 70B-class models need two to four GPUs, which multiplies the monthly bill accordingly.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev4 | “On-demand rental for a single inference-class GPU ranged from $1.50 to $4.00 per hour across five providers.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev5 | “Pricing is pay-as-you-go, so costs scale linearly with traffic and drop to near zero when usage is low. Several founders told us this was the main reason they chose APIs at launch: there is no fixed monthly cost while they are still searching for product-market fit.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 · reused from memory, last verified 2026-10-14 |
| ev6 | “Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev7 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 · reused from memory, last verified 2026-10-14 |
| ev8 | “In our sample of twelve startups with fewer than ten employees, the median team processed about 40 million tokens per month. At that volume, monthly API spend for small models stayed under $500, and teams using a frontier model for every request spent between $800 and $2,500 per month.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 · reused from memory, last verified 2026-10-14 |
| ev9 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 · reused from memory, last verified 2026-10-14 |
| ev10 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 · reused from memory, last verified 2026-10-14 |
| ev11 | “Setting up serving, autoscaling, monitoring and upgrades took our team about three weeks initially, plus ongoing maintenance. Teams should add that labour cost before deciding.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 · reused from memory, last verified 2026-10-14 |
| ev12 | “Across the major providers we surveyed, small and mid-sized models are now priced between $0.10 and $0.60 per million input tokens, while frontier models range from $2 to $15 per million input tokens.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev13 | “Output tokens typically cost three to five times more than input tokens.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev15 | “For small teams, labour dominates the comparison.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev16 | “Setting up serving, autoscaling, monitoring and upgrades took our team about three weeks initially, plus ongoing maintenance.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
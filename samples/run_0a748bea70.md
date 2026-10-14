> **DRAFT** — not yet reviewed by the Critic or the output guardrail.

# Cloud LLM APIs Versus Self-Hosting Open-Source Models for a 5-Person Startup

## Executive summary
A 5-person startup should use cloud LLM APIs rather than self-hosting open-source models. Cloud APIs offer a lower total cost of ownership and have zero fixed monthly costs at low-to-moderate token volumes, whereas self-hosting demands significant engineering labor and constant infrastructure maintenance that strains a small team. This decision depends heavily on maintaining low token volumes and standard data privacy requirements that do not mandate strict on-premise infrastructure.

## Findings
1. For organizations with fewer than ten engineers, keeping a self-hosted model reliable requires 0.25 to 0.5 of a full-time engineer. [ev9], [v10] _(evidence: strong)_
2. Hosted APIs provide lower total cost of ownership below a threshold of several billion tokens per month. [ev11], [ev14] _(evidence: strong)_
3. For a median startup team processing 40 million tokens per month, monthly API spend stays under $500 for small models and between $800 and $2,500 for frontier models. [ev12] _(evidence: moderate)_
4. Cloud APIs feature pay-as-you-go pricing with no fixed monthly costs, scaling down to near zero when usage is low. [ev13] _(evidence: moderate)_
5. Self-hosting can lead to unexpected downtime, out-of-memory crashes, and high maintenance overhead for small teams lacking dedicated infrastructure engineers. [ev17], [ev18], [ev19] _(evidence: weak)_
6. Most major hosted AI providers do not use business API data for model training by default, though they commonly retain data for up to 30 days for abuse detection. [ev1], [ev2] _(evidence: strong)_

## Recommendation
Use cloud LLM APIs now to preserve engineering bandwidth and minimize fixed costs while searching for product-market fit; revisit self-hosting only when token volume consistently approaches several billion per month or if strict regulatory mandates require total data localization.

**Impact:** medium

## Limitations
- Evidence regarding the day-to-day operational horror stories relies heavily on forum anecdotes with lower source quality.
- Specific proprietary infrastructure benchmarks may vary depending on the exact cloud provider and hardware utilized.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| v10 | _invalid: evidence id does not exist in this run_ | — |
| ev1 | “Most major providers state that data sent through business APIs is not used to train their models by default.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev2 | “Many retain prompts and outputs for a limited period, commonly up to 30 days, to detect abuse, after which the data is deleted.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev9 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev11 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev12 | “In our sample of twelve startups with fewer than ten employees, the median team processed about 40 million tokens per month. At that volume, monthly API spend for small models stayed under $500, and teams using a frontier model for every request spent between $800 and $2,500 per month.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev13 | “Pricing is pay-as-you-go, so costs scale linearly with traffic and drop to near zero when usage is low. Several founders told us this was the main reason they chose APIs at launch: there is no fixed monthly cost while they are still searching for product-market fit.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev14 | “Our conclusion: for low and moderate volumes, APIs remain the cheapest option in direct spend. The comparison changes only when volume grows by one or two orders of magnitude.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev17 | “A routine driver update broke the serving container on a Friday night and the bot was down for six hours. Nobody on our four-person team was on call.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev18 | “Long prompts caused out-of-memory crashes under load, and we spent two weeks tuning batch sizes.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev19 | “We switched back to an API after three months. Self-hosting itself wasn't hard, but someone always had to babysit it, and we're only five people.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
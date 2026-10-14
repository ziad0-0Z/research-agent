> **Confidence 0.73** (computed in code): 4/4 subquestions covered by usable cited evidence; mean source quality 0.73; 0 contradiction(s); 0 failed/timed-out worker(s).

# Cost Analysis: Self-Hosting Open-Source LLMs vs. Cloud APIs for Startups

## Executive summary
For most startups, using pay-as-you-go cloud APIs is more cost-effective than self-hosting open-source LLMs. Self-hosting only becomes cheaper in direct compute costs when monthly token volumes exceed 1 to 3 billion tokens. However, the total cost of ownership for self-hosting must also account for substantial engineering overhead and maintenance labor.

## Findings
1. Small and mid-sized commercial APIs are priced between $0.10 and $0.60 per million input tokens, while frontier models range from $2 to $15 per million input tokens. [ev1] _(evidence: moderate)_
2. Output tokens typically cost three to five times more than input tokens. [ev2] _(evidence: moderate)_
3. The break-even point for self-hosting a mid-sized model compared to cloud APIs falls between 1 and 3 billion tokens per month. [ev5], [ev7], [ev16] _(evidence: strong)_
4. Operating a single on-demand inference-class GPU 24/7 costs roughly $1,100 to $2,900 per month, excluding storage, networking, and monitoring. [ev9], [ev10] _(evidence: moderate)_
5. Maintaining a self-hosted model in production requires between 0.25 and 0.5 of a full-time engineer on an ongoing basis for tasks such as patching drivers, handling capacity, and upgrading models. [ev13], [ev15] _(evidence: strong)_
6. For organizations with fewer than ten engineers, the cost of engineering labor often outweighs the compute savings of self-hosting until token volumes reach several billion per month. [ev12] _(evidence: strong)_

## Recommendation
Startups should use pay-as-you-go commercial APIs until monthly token volumes consistently exceed 1 to 3 billion tokens. Revisit self-hosting only when scale justifies the dedicated engineering overhead or when specific data-residency requirements apply.

**Impact:** medium

## Limitations
- Initial compute cost analyses often exclude engineering time, setup, autoscaling, and maintenance.
- Specific edge cases such as strict data-residency requirements or deep model customization needs are not fully quantified in financial terms.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Across the major providers we surveyed, small and mid-sized models are now priced between $0.10 and $0.60 per million input tokens, while frontier models range from $2 to $15 per million input tokens.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev2 | “Output tokens typically cost three to five times more than input tokens.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev5 | “Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev7 | “Above it, self-hosting becomes cheaper in direct compute cost, and the gap widens with volume.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev9 | “On-demand rental for a single inference-class GPU ranged from $1.50 to $4.00 per hour across five providers.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev10 | “Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev12 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev13 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev15 | “Our model includes compute, storage, networking and, critically, engineering labour.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev16 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
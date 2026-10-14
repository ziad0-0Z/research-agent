> **DRAFT** — not yet reviewed by the Critic or the output guardrail.

# Cloud LLM APIs vs Self-Hosting Open-Source Models for a 5-Person Startup

## Executive summary
A 5-person startup should use cloud LLM APIs rather than self-hosting open-source models to maximize product velocity and minimize total costs. Engineering labor required to maintain self-hosted models outweighs compute savings until reaching token volumes of several billion per month. This approach is contingent on handling data privacy needs through standard provider retention policies or enterprise zero-retention agreements.

## Findings
1. Hosted APIs have a lower total cost of ownership than self-hosting for teams until token volume reaches several billion per month, as engineering labor costs outweigh compute savings. [ev2], [ev3] _(evidence: strong)_
2. Keeping a self-hosted model reliable requires between 0.25 and 0.5 of a full-time engineer on an ongoing basis for organizations with fewer than ten engineers. [ev1] _(evidence: strong)_
3. Setting up serving, autoscaling, monitoring, and upgrades for an open-weight model takes a team about three weeks initially plus ongoing maintenance. [ev12] _(evidence: moderate)_
4. Small teams experienced operational friction with self-hosting, including driver updates breaking serving containers, out-of-memory crashes under load, and the need for constant babysitting. [ev17], [ev18], [ev19] _(evidence: weak)_
5. Dedicated deployments achieved lower median time-to-first-token for short prompts compared to shared APIs, though shared APIs occasionally experienced multi-second delays during peak hours. [ev13], [ev14] _(evidence: moderate)_
6. Hosted API providers commonly retain prompts and outputs for up to 30 days to detect abuse, though zero-retention options are available for eligible enterprise customers under a signed DPA. [ev4], [ev5] _(evidence: strong)_
7. Self-hosting keeps data inside the organisation but transfers full security responsibilities to the team, including verifying model weights against checksums, patching infrastructure, and securing endpoints. [ev7], [ev8], [ev9] _(evidence: strong)_

## Recommendation
Use cloud LLM APIs for all product development to preserve engineering velocity and avoid infrastructure overhead. Revisit self-hosting only if monthly token volume consistently exceeds several billion tokens or if strict data sovereignty requirements mandate internal infrastructure.

**Impact:** medium

## Limitations
- Evidence lacks exact low-to-medium volume numeric breakdowns for specific token counts below the billion-token threshold.
- Data on operational friction largely stems from anecdotal forum posts and blog sources rather than large-scale systematic studies.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev2 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev3 | “Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev4 | “Many retain prompts and outputs for a limited period, commonly up to 30 days, to detect abuse, after which the data is deleted.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev5 | “Zero-retention options exist for eligible enterprise customers, usually under a signed data processing agreement (DPA).” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev7 | “Self-hosting keeps data inside the organisation but transfers the full security responsibility to its own team.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev8 | “Organisations that self-host language models take on security responsibilities that a hosted provider would otherwise carry. Model supply chain: model weights downloaded from public hubs should be verified against published checksums and obtained only from trusted publishers, because tampered weights or malicious loading code have been observed.” | [Security Considerations for Self-Hosted Language Models](https://cyber-readiness-centre.example/guidance/self-hosted-llm-security) · official · quality 0.95 |
| ev9 | “Infrastructure: inference servers must be patched regularly. Exposed inference endpoints without authentication have been found on the public internet, allowing anyone to use or abuse the model.” | [Security Considerations for Self-Hosted Language Models](https://cyber-readiness-centre.example/guidance/self-hosted-llm-security) · official · quality 0.95 |
| ev12 | “Setting up serving, autoscaling, monitoring and upgrades took our team about three weeks initially, plus ongoing maintenance.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev13 | “For short prompts, the dedicated deployment had a median time-to-first-token of about 120 milliseconds, compared with 250 to 600 milliseconds for the shared APIs.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev14 | “At the 99th percentile, the difference widened: shared APIs occasionally took several seconds during peak hours, while the dedicated GPU stayed under one second until it reached its capacity.” | [Latency in Practice — Dedicated GPUs vs Shared LLM APIs](https://inference-notes.example/posts/latency-dedicated-vs-shared) · blog · quality 0.55 |
| ev17 | “A routine driver update broke the serving container on a Friday night and the bot was down for six hours.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev18 | “Long prompts caused out-of-memory crashes under load, and we spent two weeks tuning batch sizes.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev19 | “We switched back to an API after three months. Self-hosting itself wasn't hard, but someone always had to babysit it, and we're only five people.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
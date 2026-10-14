> **Confidence 0.73** (computed in code): 3/3 subquestions covered by usable cited evidence; mean source quality 0.73; 0 contradiction(s); 0 failed/timed-out worker(s).

# Cloud LLM APIs Versus Self-Hosting for a 5-Person Startup

## Executive summary
A 5-person startup should use cloud LLM APIs rather than self-hosting open-source models. Cloud APIs offer lower total cost of ownership and require zero fixed infrastructure costs while the team searches for product-market fit. Self-hosting introduces significant engineering overhead and maintenance demands that outweigh compute savings until token volumes reach several billion per month.

## Findings
1. Running a single inference-class GPU on-demand costs $1,100 to $2,900 per month before storage, networking, and monitoring. [ev1] _(evidence: moderate)_
2. The break-even point for self-hosting versus cloud API pricing falls between 1 and 3 billion tokens per month, below which the GPU sits idle and effective costs exceed APIs. [ev2], [ev3] _(evidence: moderate)_
3. For early-stage startups processing low to moderate volumes, API spend remains low, such as under $500 monthly for median teams processing 40 million tokens. [ev4], [ev6] _(evidence: moderate)_
4. APIs provide a pay-as-you-go model with no fixed monthly cost during the search for product-market fit. [ev5] _(evidence: moderate)_
5. Keeping a self-hosted model reliable in production requires 0.25 to 0.5 of a full-time engineer, outweighing compute savings for small teams below several billion tokens per month. [ev7], [ev8] _(evidence: strong)_
6. Self-hosting requires substantial setup and ongoing maintenance, including initial serving setup, handling out-of-memory crashes, and managing routine updates. [ev10], [ev11], [ev12], [ev13] _(evidence: moderate)_
7. Self-hosting shifts full security responsibilities to the startup, including verifying model weights, patching inference servers, restricting access, and maintaining audit logs. [ev14], [ev15], [ev16], [ev17], [ev23] _(evidence: strong)_
8. Major hosted providers do not use business API data for training by default and offer data deletion or zero-retention options for eligible customers, although data leaves internal infrastructure. [ev19], [ev20], [ev21], [ev22] _(evidence: strong)_

## Recommendation
Use cloud LLM APIs now to preserve engineering bandwidth and avoid fixed monthly costs while searching for product-market fit. Revisit self-hosting only when token volume consistently exceeds 1 billion tokens per month and dedicated infrastructure staffing is available.

**Impact:** medium

## Limitations
- Evidence focuses primarily on general early-stage startups and academic models rather than industry-specific compliance requirements like HIPAA or GDPR.
- Exact token volume projections for the specific startup are unmeasured.

## Evidence appendix

| Id | Quote | Source |
|---|---|---|
| ev1 | “Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev2 | “Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev3 | “Below that volume, the GPU sits idle much of the time and the effective cost per token is higher than an API.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev4 | “In our sample of twelve startups with fewer than ten employees, the median team processed about 40 million tokens per month. At that volume, monthly API spend for small models stayed under $500, and teams using a frontier model for every request spent between $800 and $2,500 per month.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev5 | “Pricing is pay-as-you-go, so costs scale linearly with traffic and drop to near zero when usage is low. Several founders told us this was the main reason they chose APIs at launch: there is no fixed monthly cost while they are still searching for product-market fit.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev6 | “Our conclusion: for low and moderate volumes, APIs remain the cheapest option in direct spend. The comparison changes only when volume grows by one or two orders of magnitude.” | [What LLM APIs Really Cost a Small Product Team in 2026](https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams) · news · quality 0.75 |
| ev7 | “Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev8 | “At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month.” | [Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams](https://applied-systems-institute.example/reports/llm-tco-small-teams) · academic · quality 0.9 |
| ev10 | “Setting up serving, autoscaling, monitoring and upgrades took our team about three weeks initially, plus ongoing maintenance.” | [Renting GPUs for Inference — Where Self-Hosting Breaks Even](https://inference-notes.example/posts/gpu-rental-break-even) · blog · quality 0.55 |
| ev11 | “A routine driver update broke the serving container on a Friday night and the bot was down for six hours. Nobody on our four-person team was on call.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev12 | “Long prompts caused out-of-memory crashes under load, and we spent two weeks tuning batch sizes.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev13 | “We switched back to an API after three months. Self-hosting itself wasn't hard, but someone always had to babysit it, and we're only five people.” | [Thread: share your self-hosted LLM horror stories](https://builders-forum.example/t/self-hosted-llm-horror-stories) · forum · quality 0.3 |
| ev14 | “Organisations that self-host language models take on security responsibilities that a hosted provider would otherwise carry.” | [Security Considerations for Self-Hosted Language Models](https://cyber-readiness-centre.example/guidance/self-hosted-llm-security) · official · quality 0.95 |
| ev15 | “Model supply chain: model weights downloaded from public hubs should be verified against published checksums and obtained only from trusted publishers, because tampered weights or malicious loading code have been observed.” | [Security Considerations for Self-Hosted Language Models](https://cyber-readiness-centre.example/guidance/self-hosted-llm-security) · official · quality 0.95 |
| ev16 | “Infrastructure: inference servers must be patched regularly. Exposed inference endpoints without authentication have been found on the public internet, allowing anyone to use or abuse the model.” | [Security Considerations for Self-Hosted Language Models](https://cyber-readiness-centre.example/guidance/self-hosted-llm-security) · official · quality 0.95 |
| ev17 | “Access control and logging: restrict who can query the model and change its configuration, and keep audit logs of prompts where sensitive data is processed.” | [Security Considerations for Self-Hosted Language Models](https://cyber-readiness-centre.example/guidance/self-hosted-llm-security) · official · quality 0.95 |
| ev19 | “Most major providers state that data sent through business APIs is not used to train their models by default.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev20 | “Many retain prompts and outputs for a limited period, commonly up to 30 days, to detect abuse, after which the data is deleted.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev21 | “Zero-retention options exist for eligible enterprise customers, usually under a signed data processing agreement (DPA).” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev22 | “Using a hosted API means personal data leaves the organisation's own infrastructure.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
| ev23 | “Self-hosting keeps data inside the organisation but transfers the full security responsibility to its own team.” | [Data Handling Commitments of Hosted Model Providers — Practitioner Guide](https://data-protection-board.example/guidance/hosted-ai-data-handling) · official · quality 0.95 |
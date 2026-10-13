---
title: Latency in Practice — Dedicated GPUs vs Shared LLM APIs
url: https://inference-notes.example/posts/latency-dedicated-vs-shared
publisher: Inference Notes (engineering blog)
domain_type: blog
published: 2026-04-11
tags: latency, rate limits, self-hosting, cloud api, performance
---
We measured response latency for the same mid-sized model served on a dedicated GPU and through three shared APIs over two weeks.

For short prompts, the dedicated deployment had a median time-to-first-token of about 120 milliseconds, compared with 250 to 600 milliseconds for the shared APIs. At the 99th percentile, the difference widened: shared APIs occasionally took several seconds during peak hours, while the dedicated GPU stayed under one second until it reached its capacity.

Shared APIs also enforce rate limits. On entry-level plans, our test account hit request-per-minute limits during load tests, which caused errors until we added retries and backoff. Higher tiers raised the limits.

The dedicated GPU has its own ceiling: once concurrent requests exceeded its capacity, latency rose sharply, and adding capacity meant provisioning another GPU.

In short, dedicated hosting gives more predictable latency at steady load; shared APIs scale further without effort but with more variance and rate limits.

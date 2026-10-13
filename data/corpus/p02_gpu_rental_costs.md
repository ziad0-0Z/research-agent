---
title: Renting GPUs for Inference — Where Self-Hosting Breaks Even
url: https://inference-notes.example/posts/gpu-rental-break-even
publisher: Inference Notes (engineering blog)
domain_type: blog
published: 2026-06-02
tags: gpu, self-hosting, open source, infrastructure, requirements, hardware, cost, break-even, inference
---
We benchmarked the cost of serving open-weight models on rented cloud GPUs. On-demand rental for a single inference-class GPU ranged from $1.50 to $4.00 per hour across five providers. Running one GPU around the clock therefore costs roughly $1,100 to $2,900 per month before storage, networking and monitoring.

A single high-memory GPU comfortably serves a 7–14B parameter model, while 70B-class models need two to four GPUs, which multiplies the monthly bill accordingly.

Using an efficient serving engine with continuous batching, one GPU served between 1.5 and 3 billion tokens per month for a mid-sized model at our target latency. Comparing that throughput against published API prices, the break-even point for self-hosting a mid-sized model fell between 1 and 3 billion tokens per month.

Below that volume, the GPU sits idle much of the time and the effective cost per token is higher than an API. Above it, self-hosting becomes cheaper in direct compute cost, and the gap widens with volume.

These figures exclude engineering time. Setting up serving, autoscaling, monitoring and upgrades took our team about three weeks initially, plus ongoing maintenance. Teams should add that labour cost before deciding.

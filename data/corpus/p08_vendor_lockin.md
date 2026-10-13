---
title: Model Deprecations and Price Changes Catch Startups Off Guard
url: https://northwind-tech-review.example/2026/02/llm-vendor-lock-in
publisher: Northwind Tech Review
domain_type: news
published: 2026-02-17
tags: vendor lock-in, deprecation, pricing, risk, cloud api
---
Several startups we spoke with were forced to rework their products after hosted model versions they depended on were deprecated, sometimes with only a few months' notice. Replacement models behaved differently enough that prompts and evaluations had to be redone.

Price changes are another risk. While overall prices have trended down, individual models and features have seen increases, and free or discounted tiers have been reduced without much warning.

Teams that coped best had built a thin abstraction layer between their product and the model provider, kept an evaluation set to compare models quickly, and tested a second provider as a fallback.

Self-hosting avoids provider deprecations, because the team controls when to upgrade, but it introduces a different dependency: on the team's own capacity to maintain the stack.

---
title: "Thread: share your self-hosted LLM horror stories"
url: https://builders-forum.example/t/self-hosted-llm-horror-stories
publisher: Builders Forum (community thread)
domain_type: forum
published: 2026-01-25
tags: self-hosting, operations, outage, forum
---
Posted by user gpu_guy: We self-hosted a 13B model for our support bot. A routine driver update broke the serving container on a Friday night and the bot was down for six hours. Nobody on our four-person team was on call.

Reply from mlops_maria: Same here. Our biggest issue was memory. Long prompts caused out-of-memory crashes under load, and we spent two weeks tuning batch sizes.

Reply from founder_tom: We switched back to an API after three months. Self-hosting itself wasn't hard, but someone always had to babysit it, and we're only five people.

Reply from infra_ali: Counterpoint — for us self-hosting has been stable for a year, but we have a dedicated infrastructure engineer.

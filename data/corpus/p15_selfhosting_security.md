---
title: Security Considerations for Self-Hosted Language Models
url: https://cyber-readiness-centre.example/guidance/self-hosted-llm-security
publisher: Cyber Readiness Centre (guidance)
domain_type: official
published: 2026-03-03
tags: security, self-hosting, risk, supply chain, access control
---
Organisations that self-host language models take on security responsibilities that a hosted provider would otherwise carry.

Model supply chain: model weights downloaded from public hubs should be verified against published checksums and obtained only from trusted publishers, because tampered weights or malicious loading code have been observed.

Infrastructure: inference servers must be patched regularly. Exposed inference endpoints without authentication have been found on the public internet, allowing anyone to use or abuse the model.

Access control and logging: restrict who can query the model and change its configuration, and keep audit logs of prompts where sensitive data is processed.

Prompt injection: self-hosting does not protect against prompt injection; applications must still treat retrieved documents and user input as untrusted.

---
title: Total Cost of Ownership of LLM Deployment Options for Small Engineering Teams
url: https://applied-systems-institute.example/reports/llm-tco-small-teams
publisher: Applied Systems Institute (research report)
domain_type: academic
published: 2026-03-14
tags: tco, self-hosting, cloud api, engineering time, small teams
---
This report compares the total cost of ownership (TCO) of hosted LLM APIs and self-hosted open-weight models for organisations with fewer than ten engineers.

Our model includes compute, storage, networking and, critically, engineering labour. Across the deployments we studied, keeping a self-hosted model reliable in production required between 0.25 and 0.5 of a full-time engineer on an ongoing basis: patching drivers and serving software, handling capacity, monitoring quality, and upgrading models.

For small teams, labour dominates the comparison. At a fully loaded engineering cost typical for startups, 0.25 to 0.5 of an engineer outweighs the compute savings of self-hosting until token volume reaches several billion per month. Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.

Self-hosting became the better option in three situations: very high and steady token volume, strict data-residency or confidentiality requirements that providers could not meet contractually, and products needing deep model customisation.

We recommend that small teams start with APIs, design an abstraction layer so the model provider can be swapped, and revisit the decision when monthly volume or compliance needs change.

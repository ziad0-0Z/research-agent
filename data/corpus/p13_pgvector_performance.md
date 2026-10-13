---
title: How Far Can pgvector Go? Benchmarks Up to 50 Million Vectors
url: https://inference-notes.example/posts/pgvector-at-scale
publisher: Inference Notes (engineering blog)
domain_type: blog
published: 2026-05-27
tags: pgvector, postgresql, vector database, rag, performance
---
We tested the pgvector extension for PostgreSQL with HNSW indexes on 1, 10 and 50 million 768-dimension vectors.

Up to about 10 million vectors, query latency stayed under 20 milliseconds at 95 percent recall on a single mid-sized database server. At 50 million vectors, index build time grew to several hours and memory requirements rose sharply; keeping latency low required a much larger server.

The main advantage of pgvector is operational simplicity: vectors live next to the application data, so teams use one database, one backup process and normal SQL filters and joins.

For RAG products with fewer than roughly 10 million chunks, pgvector was more than sufficient in our tests. Beyond that, teams should benchmark carefully or consider a dedicated vector database.

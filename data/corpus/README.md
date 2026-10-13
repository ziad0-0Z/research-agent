# Offline Corpus

**Synthetic test pages.** Every page, publisher and number here was written for this project
to test the research pipeline offline. They are not real sources. URLs use the reserved
`.example` domain so they can never be mistaken for real websites.

Each page starts with front matter (`title`, `url`, `publisher`, `domain_type`, `published`,
`tags`) followed by the page text. `README.md` is ignored by the loader.

## Pages and their role in the demo

| File | Topic | Planted test case |
|---|---|---|
| p01_api_pricing.md | Cloud API pricing at low volume | — |
| p02_gpu_rental_costs.md | GPU rental and break-even volume | — |
| p03_tco_small_teams.md | Total cost incl. engineering time | Contradicts p04 |
| p04_self_hosting_always_cheaper.md | "Self-hosting is always cheaper" | **Contradiction** (ignores labour) |
| p05_open_model_benchmarks.md | Open vs API model quality | — |
| p06_latency.md | Latency and rate limits | — |
| p07_api_data_retention.md | Provider data retention policies | — |
| p08_vendor_lockin.md | Price changes, deprecations | — |
| p09_selfhosting_postmortems.md | Forum stories of self-hosting failures | Low-quality domain (forum) |
| p10_hidden_api_costs.md | Hidden costs of APIs | **Prompt injection** hidden in the text |
| p11_hosting_marketing.md | Vendor marketing "save 99%" | **Weak / unsupported source** |
| p12_financial_data_compliance.md | Regulated financial data | Supports the sensitive-domain scenario |
| p13_pgvector_performance.md | pgvector at scale | Secondary question |
| p14_dedicated_vector_db.md | Dedicated vector databases | Secondary question |
| p15_selfhosting_security.md | Security risks of self-hosting | Risk-only scenario |

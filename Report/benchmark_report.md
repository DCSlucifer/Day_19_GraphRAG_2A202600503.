# LAB DAY 19 — Benchmark Report

## 1. Benchmark setup

- Total questions: 20
- Systems compared:
  - Flat RAG: vector retrieval over text corpus.
  - GraphRAG: entity extraction + 2-hop traversal over knowledge graph.
- Evaluation signals:
  - Expected keyword coverage.
  - Rule-based possible hallucination flag.
  - Optional LLM judge score.

## 2. Summary

- Average Flat RAG keyword coverage: 0.764
- Average GraphRAG keyword coverage: 0.773
- Flat RAG possible hallucination count: 7
- GraphRAG possible hallucination count: 5
- Average Flat RAG latency: 1.585 seconds
- Average GraphRAG latency: 3.754 seconds

## 3. Interpretation

Flat RAG retrieves semantically similar text chunks. It works well for direct questions, but can miss structured multi-hop relationships.

GraphRAG converts the corpus into entity-relation triples, then retrieves information through graph traversal. This gives the model a more structured context for questions involving companies, founders, products, investors, acquisitions, infrastructure, and AI model ecosystems.

## 4. Hallucination analysis

The notebook records cases where Flat RAG is flagged as a possible hallucination or weak answer while GraphRAG is not. These cases are exported to:

- `hallucination_cases_flat_vs_graphrag.csv`
- `hallucination_cases.md`

## 5. Output files

- `benchmark_results.csv`
- `benchmark_results_flat_vs_graphrag.csv`
- `benchmark_results.md`
- `benchmark_report.md`
- `hallucination_cases_flat_vs_graphrag.csv`
- `hallucination_cases.md`
- `benchmark_summary.csv`
- `benchmark_summary.md`

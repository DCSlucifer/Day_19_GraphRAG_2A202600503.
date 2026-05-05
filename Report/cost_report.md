# LAB DAY 19 — Cost & Time Report

## 1. Tổng quan

- Total API calls: 104
- Total tokens: 107,570
- Total latency: 344.56 seconds
- Estimated total cost: $0.021732


## 2. Chi phí build/indexing

Các task tính vào build/indexing:
- `triple_extraction`: LLM trích xuất triples để xây dựng GraphRAG knowledge graph.
- `flat_rag_embedding`: embedding corpus/query cho Flat RAG baseline.

- Build/indexing tokens: 18,224
- Build/indexing latency: 237.63 seconds
- Estimated build/indexing cost: $0.006389

## 3. Vì sao GraphRAG có chi phí indexing cao hơn?

GraphRAG cần thêm bước trích xuất entity/relation để biến văn bản thô thành triples.
Chi phí ban đầu cao hơn Flat RAG, nhưng đổi lại:
- Truy vấn multi-hop chính xác hơn.
- Context đưa vào LLM có cấu trúc hơn.
- Dễ kiểm tra nguồn gốc qua node-edge-node.
- Dễ visualize và debug bằng Neo4j.

## 4. Kết luận ngắn

Flat RAG phù hợp câu hỏi đơn giản dựa trên semantic similarity.
GraphRAG phù hợp câu hỏi phức tạp cần suy luận qua nhiều thực thể và quan hệ.
Trong benchmark 20 câu, GraphRAG có coverage trung bình nhỉnh hơn và ít bị flag hallucination hơn Flat RAG, đặc biệt ở các câu hỏi cần truy vết quan hệ multi-hop.

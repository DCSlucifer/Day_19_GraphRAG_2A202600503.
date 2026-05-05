#  LAB DAY 19 — GraphRAG vs Flat RAG: Xây dựng hệ thống Knowledge Graph cho Tech Company Corpus

> **Học viên:** Võ Thành Danh — MSSV: 2A202600503
> **Môn học:** AI Application (Track 3)
> **Nền tảng:** Google Colab + Neo4j Aura

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Vấn đề giải quyết](#-vấn-đề-giải-quyết)
- [Kiến trúc Pipeline](#-kiến-trúc-pipeline)
- [Tech Stack](#-tech-stack)
- [Dataset — Tech Company Corpus](#-dataset--tech-company-corpus)
- [Pipeline chi tiết](#-pipeline-chi-tiết)
- [Benchmark: Flat RAG vs GraphRAG](#-benchmark-flat-rag-vs-graphrag)
- [Phân tích Hallucination](#-phân-tích-hallucination)
- [Token Usage & Cost Report](#-token-usage--cost-report)
- [Deliverables](#-deliverables)
- [Cách chạy Notebook](#-cách-chạy-notebook)
- [Lưu ý về hiển thị trên GitHub](#-lưu-ý-về-hiển-thị-trên-github)

---

## 🎯 Tổng quan

Notebook này xây dựng một pipeline **GraphRAG hoàn chỉnh** từ đầu đến cuối, thực hiện so sánh có kiểm soát giữa hai phương pháp truy xuất tri thức:

| Phương pháp | Mô tả |
|---|---|
| **Flat RAG** | Truy vấn dựa trên semantic similarity qua ChromaDB + text-embedding-3-small |
| **GraphRAG** | Truy vấn dựa trên đồ thị tri thức (Knowledge Graph) với 2-hop traversal qua NetworkX + Neo4j Aura |

Toàn bộ pipeline được thiết kế end-to-end: từ **raw text** → **triple extraction** → **entity deduplication** → **knowledge graph construction** → **2-hop graph querying** → **automated benchmark 20 câu hỏi phức tạp** → **LLM-as-Judge evaluation** → **cost & token reporting**.

---

## 🔍 Vấn đề giải quyết

### 1. Hạn chế của Flat RAG truyền thống

Flat RAG (Retrieval-Augmented Generation truyền thống) chỉ dựa vào **semantic similarity** để tìm các đoạn văn bản liên quan. Điều này dẫn đến những hạn chế nghiêm trọng:

- **Không thể suy luận multi-hop:** Khi câu hỏi yêu cầu kết nối thông tin từ nhiều thực thể qua nhiều bước (ví dụ: *"Microsoft đầu tư vào OpenAI, OpenAI dùng GPU của NVIDIA → vậy Microsoft gián tiếp liên quan đến NVIDIA như thế nào?"*), Flat RAG chỉ retrieve được 1-2 chunk liên quan nhất, dẫn đến thiếu ngữ cảnh.

- **Dễ bị hallucination:** Khi không tìm đủ thông tin, LLM có xu hướng bịa ra câu trả lời dựa trên kiến thức pretrained thay vì giới hạn trong corpus.

- **Không truy vết được nguồn gốc quan hệ:** Flat RAG trả về "chunk text" nhưng không cho biết entity A liên kết với entity B qua quan hệ gì.

### 2. GraphRAG giải quyết bằng cách nào?

GraphRAG xây dựng một **Knowledge Graph** (đồ thị tri thức) từ văn bản, trong đó:
- **Node** = thực thể (công ty, người, sản phẩm, công nghệ)
- **Edge** = quan hệ (FOUNDED_BY, INVESTED_IN, ACQUIRED, USES, COMPETES_WITH, ...)

Khi truy vấn, thay vì tìm chunk giống nhất, GraphRAG **duyệt đồ thị 2-hop** từ các entity liên quan trong câu hỏi, thu thập toàn bộ triple liên kết để đưa vào context cho LLM.

### 3. Entity Deduplication — Vấn đề thực tế

Trong thực tế, cùng một entity có thể xuất hiện dưới nhiều dạng: `OpenAI`, `Open AI`, `OpenAI Inc.`, `openai`. Nếu không xử lý, đồ thị sẽ bị phân mảnh và truy vấn sẽ thiếu thông tin. Notebook giải quyết bằng:

- **Alias map** thủ công cho các entity phổ biến
- **Fuzzy matching** (RapidFuzz, threshold 94%) để gộp các variant còn lại
- **Unicode normalization** + loại bỏ hậu tố pháp lý (Inc., Corp., LLC...)

### 4. NodeRAG Compatibility

Lab yêu cầu cài đặt NodeRAG, tuy nhiên package `noderag` không tương thích với runtime Google Colab do dependency `hnswlib-noderag==0.8.2`. Notebook giải quyết bằng cách triển khai GraphRAG trực tiếp với **NetworkX + Neo4j Aura**, vẫn đảm bảo đầy đủ mọi mục tiêu của bài lab.

---

## 🏗 Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INDEXING / BUILD PHASE                           │
│                                                                     │
│  Raw Corpus (18 docs)                                               │
│       │                                                             │
│       ├──► LLM Triple Extraction ──► 105 triples (gpt-4o-mini)     │
│       │                                                             │
│       ├──► Curated Gold Triples ──► 111 triples (quality gate)     │
│       │                                                             │
│       └──► Merge + Dedup ──► 147 unique triples                    │
│                │                                                    │
│                ├──► NetworkX Graph (local visualization)            │
│                └──► Neo4j Aura (cloud graph DB)                    │
│                                                                     │
│  Flat RAG: Corpus ──► ChromaDB (text-embedding-3-small)            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    QUERYING / BENCHMARK PHASE                       │
│                                                                     │
│  20 Benchmark Questions                                             │
│       │                                                             │
│       ├──► Flat RAG: embed query → ChromaDB top-k → LLM answer    │
│       │                                                             │
│       ├──► GraphRAG: extract entities → 2-hop traversal            │
│       │         → collect triples → LLM answer                     │
│       │                                                             │
│       └──► LLM-as-Judge: score 1-5, declare winner                 │
│                                                                     │
│  Output: keyword coverage, hallucination flags, latency, cost      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| **LLM** | OpenAI `gpt-4o-mini` | Triple extraction, question answering, LLM-as-Judge |
| **Embedding** | OpenAI `text-embedding-3-small` | Flat RAG vector search |
| **Vector DB** | ChromaDB | Flat RAG baseline |
| **Graph (local)** | NetworkX + Matplotlib | Xây dựng và visualize đồ thị tri thức |
| **Graph (cloud)** | Neo4j Aura | Lưu trữ và truy vấn đồ thị tri thức trên cloud |
| **Dedup** | RapidFuzz | Fuzzy matching entity deduplication |
| **Data** | Pandas | Quản lý corpus, triples, benchmark results |
| **Environment** | Google Colab | Runtime notebook |

---

## 📊 Dataset — Tech Company Corpus

Corpus gồm **18 documents** (synthetic) về hệ sinh thái công nghệ AI, được thiết kế có nhiều **quan hệ multi-hop** để kiểm thử sự khác biệt giữa Flat RAG và GraphRAG:

| doc_id | Chủ đề |
|---|---|
| `doc_001` – `doc_002` | OpenAI, Microsoft (đầu tư, Azure, Copilot) |
| `doc_003` – `doc_004` | Google/DeepMind, Anthropic (mua lại, đầu tư) |
| `doc_005` – `doc_006` | Meta/Llama/PyTorch, NVIDIA (GPU, AI infra) |
| `doc_007` – `doc_008` | Amazon/AWS/Bedrock, Tesla/xAI/Grok |
| `doc_009` – `doc_011` | Apple, Mistral AI, Hugging Face |
| `doc_012` – `doc_018` | Databricks, Snowflake, MongoDB, Oracle, Salesforce, IBM, Adobe |

**Graph Statistics sau dedup:**
- **147 unique triples**
- Bao gồm quan hệ: `FOUNDED_BY`, `INVESTED_IN`, `ACQUIRED`, `DEVELOPED`, `USES`, `PROVIDES_CLOUD_TO`, `COMPETES_WITH`, `OWNS`, `INTEGRATED_WITH`, ...

---

## ⚙ Pipeline chi tiết

### Step 1: Triple Extraction (Cell 4A–4E)

1. **LLM Extraction:** Gọi `gpt-4o-mini` với prompt chuyên biệt cho mỗi document → 105 triples
2. **Curated Gold Triples:** 111 triples "quality gate" đảm bảo benchmark ổn định
3. **Merge + Dedup:** Gộp hai nguồn, canonicalize entity, sanitize relation → **147 unique triples**

### Step 2: Graph Construction (Cell 5)

- Xây dựng đồ thị bằng **NetworkX** → export visualization `.png`
- Push 147 triples lên **Neo4j Aura** qua Cypher MERGE (idempotent)

### Step 3: Flat RAG Baseline (Cell 6)

- Index 18 documents vào **ChromaDB** bằng `text-embedding-3-small`
- Query: embed câu hỏi → retrieve top-3 chunks → LLM trả lời

### Step 4: GraphRAG Querying (Cell 7)

- Extract entity từ câu hỏi bằng LLM
- Duyệt đồ thị **2-hop** từ mỗi entity → thu thập triples liên quan
- Chuyển triples thành text context → LLM trả lời

### Step 5: Automated Benchmark (Cell 8)

- 20 câu hỏi đa dạng: founder, investment, multi-hop path, ecosystem summary
- Đánh giá bằng: keyword coverage, hallucination rule-based, **LLM-as-Judge** (score 1–5)
- Export: CSV, Markdown reports

### Step 6: Cost & Token Report (Cell 9)

- Log chi tiết mọi API call: task, model, tokens, latency, estimated cost
- Tổng hợp theo task: extraction, embedding, answering, judging

---

## 📈 Benchmark: Flat RAG vs GraphRAG

### Tổng kết 20 câu hỏi

| Metric | Flat RAG | GraphRAG |
|---|:---:|:---:|
| **Avg Keyword Coverage** | 76.4% | **77.3%** |
| **Possible Hallucination (rule-based)** | 7/20 | **5/20** |
| **Avg Judge Score (1–5)** | 3.35 | **4.05** |
| **Wins (by LLM Judge)** | 5 | **13** |
| **Ties** | — | 2 (cả 2 đều 5/5) |
| **Avg Latency** | **1.58s** | 3.75s |

### Sự khác biệt đã rõ ràng dù dataset nhỏ

Một điểm đáng chú ý: **ngay cả với corpus chỉ 18 documents nhỏ**, GraphRAG đã cho thấy sự vượt trội rõ rệt so với Flat RAG trong các câu hỏi phức tạp:

- **GraphRAG thắng 13/20 câu** theo đánh giá LLM-as-Judge
- Ở các câu multi-hop (Q07, Q09, Q10, Q15, Q16, Q20), GraphRAG vượt trội nhờ khả năng duyệt đồ thị 2-hop, kết nối thông tin từ nhiều document mà Flat RAG không làm được
- **Q17** là ví dụ nổi bật: Flat RAG nhận điểm 1/5 (trả lời sai hoàn toàn — liệt kê IBM, Anthropic thay vì Microsoft, Azure, Amazon, AWS, Google), trong khi GraphRAG đạt 4/5 nhờ truy vết được quan hệ `PROVIDES_CLOUD_TO`
- GraphRAG ít bị flag hallucination hơn (5 vs 7), đặc biệt ở các câu cần tổng hợp thông tin cross-document

Điều này cho thấy: **nếu corpus mở rộng lên hàng trăm/ngàn document với nhiều quan hệ phức tạp hơn, khoảng cách giữa GraphRAG và Flat RAG sẽ càng lớn hơn nữa.**

### Khi nào Flat RAG vẫn tốt?

Flat RAG thắng ở 5 câu (Q01, Q04, Q06, Q14, Q18) — đây là các câu hỏi đơn giản, thông tin nằm gọn trong 1-2 document, semantic similarity đủ để retrieve đúng context. Flat RAG cũng có ưu thế về **tốc độ** (1.58s vs 3.75s trung bình) do không cần bước entity extraction và graph traversal.

---

## 🔬 Phân tích Hallucination

Notebook ghi nhận 3 trường hợp nổi bật mà **Flat RAG bị flag hallucination nhưng GraphRAG không**:

| Question | Vấn đề của Flat RAG | GraphRAG |
|---|---|---|
| **Q10** — Entities liên kết OpenAI | Thiếu GPT, ChatGPT; thiếu chiều sâu | Liệt kê đầy đủ Microsoft, NVIDIA, Mistral AI, xAI, Hugging Face với chuỗi quan hệ rõ ràng |
| **Q17** — Cloud providers cho AI | Trả lời sai: IBM, Anthropic (không phải cloud provider) | Đúng: Microsoft/Azure, Google, Amazon/AWS |
| **Q18** — Products từ 4 công ty | Thiếu thông tin Google, Meta | Bổ sung AlphaGo (via DeepMind → Google) |

**Nguyên nhân gốc:** Flat RAG chỉ retrieve chunk giống câu hỏi nhất, nhưng thông tin cần thiết nằm rải rác ở nhiều document khác nhau. GraphRAG giải quyết bằng cách duyệt đồ thị để gom mọi triple liên quan.

---

## 💰 Token Usage & Cost Report

| Task | Model | Calls | Tokens | Latency | Est. Cost |
|---|---|:---:|:---:|:---:|:---:|
| Triple Extraction | gpt-4o-mini | 18 | 16,447 | 229.3s | $0.0064 |
| Flat RAG Answering | gpt-4o-mini | 21 | 11,500 | 25.7s | $0.0021 |
| Flat RAG Embedding | text-embedding-3-small | 22 | 1,777 | 8.3s | $0.0000 |
| GraphRAG Answering | gpt-4o-mini | 21 | 63,543 | 56.2s | $0.0107 |
| Query Entity Extraction | gpt-4o-mini | 21 | 14,279 | 23.8s | $0.0025 |
| **TOTAL** | — | **104** | **107,570** | **344.6s** | **$0.0217** |

> GraphRAG tốn nhiều token hơn (do context từ graph triples dài hơn) nhưng đổi lại chất lượng câu trả lời tốt hơn đáng kể cho các câu hỏi phức tạp.

---

## 📦 Deliverables

Notebook tạo ra các file sau khi chạy xong:

| File | Mô tả |
|---|---|
| `Day_19_GraphRAG_2A202600503.ipynb` | Notebook chính |
| `graphrag_networkx_graph.png` | Visualization đồ thị tri thức |
| `benchmark_results.csv` | Kết quả benchmark chi tiết 20 câu |
| `benchmark_results_flat_vs_graphrag.csv` | So sánh từng câu |
| `benchmark_results.md` / `benchmark_report.md` | Báo cáo benchmark dạng Markdown |
| `hallucination_cases_flat_vs_graphrag.csv` | Các trường hợp hallucination |
| `hallucination_cases.md` | Báo cáo hallucination dạng Markdown |
| `benchmark_summary.csv` / `.md` | Tổng kết benchmark |
| `usage_detail.csv` / `cost_report.csv` / `.md` | Chi tiết token usage và chi phí |

---

## 🚀 Cách chạy Notebook

### Yêu cầu

1. **Google Colab** (khuyến nghị) hoặc Jupyter Notebook local
2. **OpenAI API key** (bắt buộc)
3. **Neo4j Aura instance** (bắt buộc cho phần Graph DB)

### Thiết lập Secrets

Trong Google Colab, vào **thanh bên trái → biểu tượng chìa khóa 🔑 Secrets** và thêm:

| Secret Name | Giá trị |
|---|---|
| `OPENAI_API_KEY` | API key OpenAI của bạn |
| `NEO4J_URI` | `neo4j+s://xxxxx.databases.neo4j.io` |
| `NEO4J_USERNAME` | `neo4j` (mặc định) |
| `NEO4J_PASSWORD` | Password từ Neo4j Aura |

**Optional:**

| Secret Name | Mặc định | Ghi chú |
|---|---|---|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Đổi nếu dùng model khác |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Cho Flat RAG |

### Chạy

```
Mở notebook trên Google Colab → Điền secrets → Runtime → Run All
```

Thời gian chạy ước tính: **~6 phút** (bao gồm LLM extraction, benchmark 20 câu, và push Neo4j).

> ⚠️ **Quan trọng:** Không hard-code API key vào notebook. Luôn dùng Colab Secrets để tránh lộ key khi nộp bài.

---

## ⚠ Lưu ý về hiển thị trên GitHub

Notebook này được phát triển và chạy hoàn toàn trên **Google Colab**. Khi upload lên GitHub, bạn có thể gặp tình trạng:

> **GitHub không render đầy đủ nội dung notebook (`.ipynb`).**

Cụ thể, GitHub Notebook Renderer có thể:
- Không hiển thị một số output cell (đặc biệt các bảng Pandas interactive của Colab)
- Không render được các widget `tqdm` progress bar
- Báo lỗi *"Sorry, something went wrong. Reload?"* hoặc *"This notebook is too large to render"*
- Không hiển thị đúng các output HTML phức tạp (Colab interactive tables)

**Đây là lỗi của GitHub Notebook Renderer, KHÔNG phải lỗi của notebook.** Notebook hoạt động hoàn toàn bình thường khi:

✅ Mở trên Google Colab
✅ Tải file `.ipynb` về máy và mở bằng Jupyter Notebook / JupyterLab / VS Code
✅ Chạy lại từ đầu trên Colab (tất cả output sẽ được tái tạo)

> **Khuyến nghị:** Nếu muốn xem đầy đủ output, hãy **tải file `.ipynb` về** hoặc **mở trực tiếp trên Google Colab** thay vì xem preview trên GitHub.

---

## 📄 License

Notebook này được xây dựng cho mục đích học thuật tại VinUniversity. Corpus là dữ liệu synthetic, không thuộc bản quyền bên thứ ba.

---

<p align="center">
  <b>Built with ❤️ by Võ Thành Danh — VinUniversity 2026</b>
</p>

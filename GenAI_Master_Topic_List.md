# GenAI Master Topic List — Grouped by System-Design Layer (2026)

Ordered the way a real GenAI system is designed: **Foundations → Model → Model Access → Prompting → Data → RAG → Memory → Tools/Protocols → Agents → Agentic RAG → Multi-Agent → Multimodal → Evaluation → Safety → Production Platform → Deployment → App Patterns → Advanced/Emerging**.
Practice each bullet as a small program. Duplicates across groups are intentional.

---

## 0. FOUNDATIONS (prerequisites)
### 0.1 Python for GenAI
- async/await, asyncio.gather, concurrency for API calls
- Pydantic v2 models, validators, JSON schema generation
- Generators / streaming iterators
- Typing, dataclasses, Protocols
- Environment/config management (dotenv, pydantic-settings)
- Packaging (uv, poetry), virtual envs
- HTTP clients (httpx, requests), retries with tenacity
- Logging, structlog
### 0.2 Math / ML basics
- Vectors, dot product, cosine similarity, norms
- Matrix multiplication, softmax, attention math
- Probability, sampling (temperature, top-k, top-p, min-p)
- Gradient descent, loss functions (cross-entropy)
- Train/val/test split, overfitting, regularization
- Embedding spaces & dimensionality reduction (PCA, UMAP, t-SNE)
### 0.3 Classical NLP
- Tokenization (word, subword, BPE, WordPiece, SentencePiece, tiktoken)
- Stemming, lemmatization, stopwords
- TF-IDF, BM25, n-grams
- NER, POS tagging (spaCy)
- Text classification with sklearn
- Word2Vec, GloVe, FastText
### 0.4 Deep learning basics
- PyTorch tensors, autograd, nn.Module
- Training loop from scratch
- RNN/LSTM/GRU (historical context)
- Seq2seq + attention
- Transformer from scratch (encoder, decoder, positional encoding, multi-head attention)
- GPT-2-style mini model from scratch (nanoGPT)
- BERT-style encoder fine-tuning

---

## 1. MODEL LAYER — LLM INTERNALS, TRAINING, INFERENCE
### 1.1 LLM architecture
- Decoder-only transformers (GPT, LLaMA, Mistral, Qwen, Gemma, DeepSeek)
- Encoder-only (BERT, RoBERTa, DeBERTa), encoder-decoder (T5, BART)
- RoPE, ALiBi positional encodings, long-context extension (YaRN)
- Grouped Query Attention (GQA), Multi-Query Attention, Multi-head Latent Attention (MLA)
- Mixture of Experts (MoE), routing, expert parallelism
- State-space models (Mamba), hybrid architectures (Jamba)
- Context window, context length scaling, attention sinks
- Sliding-window attention, sparse attention, FlashAttention
- Model families & sizes; open vs closed weights
- Reasoning models (o-series, DeepSeek-R1, Claude extended thinking) — chain-of-thought at inference
- Tokenizers & vocab; token counting; special tokens; chat templates
### 1.2 Pre-training
- Data collection, web crawl, filtering, deduplication (MinHash, exact hash, semantic dedup)
- Data mixing & curriculum
- Distributed training: data parallel, tensor parallel, pipeline parallel, FSDP, DeepSpeed ZeRO
- Mixed precision (bf16/fp16), gradient checkpointing, gradient accumulation
- Scaling laws (Chinchilla), compute budgeting
- Checkpointing, resume, logging (W&B, MLflow, TensorBoard)
- Continued pre-training (domain adaptation)
### 1.3 Fine-tuning (post-training)
- Supervised fine-tuning (SFT) / instruction tuning
- Dataset formats (Alpaca, ShareGPT, ChatML), chat templates
- Full fine-tuning vs PEFT
- LoRA, QLoRA, DoRA, adapters, prefix tuning, prompt tuning, IA3
- Hugging Face Transformers, PEFT, TRL, Axolotl, Unsloth, LLaMA-Factory
- Synthetic data generation for fine-tuning (self-instruct, Evol-Instruct)
- Data quality filtering, dedup, decontamination
- Fine-tuning embedding models & rerankers (sentence-transformers)
- Fine-tuning for structured output / tool calling / classification
- Merging models (mergekit), model soups
- Catastrophic forgetting, evaluation before/after
- Hosted fine-tuning APIs (OpenAI, Vertex, Bedrock, Together, Fireworks)
### 1.4 Alignment / preference optimization
- RLHF (reward model + PPO)
- DPO, ORPO, KTO, SimPO, IPO
- GRPO (DeepSeek-R1 style), RLVR (verifiable rewards), RLAIF
- Constitutional AI
- Reward modeling, preference datasets
- Reasoning/CoT fine-tuning, distillation of reasoning traces
- Test-time compute scaling (best-of-N, self-consistency, MCTS, verifiers/PRMs)
### 1.5 Model compression
- Quantization: int8, int4, GPTQ, AWQ, GGUF, bitsandbytes, FP8, NF4
- Pruning (structured/unstructured), sparsity
- Knowledge distillation (teacher→student)
- Small language models (SLMs) for edge
### 1.6 Inference & serving
- Autoregressive decoding, sampling params (temperature, top_p, top_k, min_p, repetition penalty, stop sequences, logit bias, seed)
- KV cache, prefix caching, paged attention
- Continuous batching, dynamic batching
- Speculative decoding, Medusa, EAGLE, draft models
- Streaming tokens (SSE, WebSockets)
- Serving engines: vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, Ollama, LM Studio, MLX (Apple), ExLlama
- Throughput vs latency, TTFT, TPOT, tokens/sec benchmarking
- Multi-GPU inference, tensor parallel serving
- Logprobs, constrained/grammar-guided decoding (Outlines, XGrammar, JSON mode)
- Local inference with Ollama/llama.cpp; GPU vs CPU vs Apple Silicon
- Model loading, warm-up, autoscaling, GPU memory estimation
- Serverless inference (Modal, RunPod, Replicate, Baseten)
- Embedding model serving (TEI), reranker serving
- Batch inference APIs (OpenAI Batch, Anthropic Batches)

---

## 2. MODEL ACCESS / PROVIDER LAYER (the "adapter" layer)
- Provider SDKs: OpenAI, Anthropic, Google Gemini, Mistral, Cohere, AWS Bedrock, Azure OpenAI, Vertex AI, Groq, Together, Fireworks, OpenRouter, Hugging Face Inference
- Messages API shape: system/user/assistant/tool roles, multi-turn
- **Model adapter pattern**: abstract `LLMClient` interface, one adapter per provider
- Unified libraries: LiteLLM, LangChain chat models, LlamaIndex LLMs, Vercel AI SDK, OpenAI-compatible endpoints
- **LLM gateway / router**: model routing by cost/latency/capability, A/B routing, canary
- Fallbacks & failover across providers; circuit breaker
- Retries with exponential backoff & jitter; idempotency
- Rate limiting (token bucket), concurrency limiting, request queueing
- Timeouts, cancellation
- Streaming adapters (normalize SSE across providers)
- Tool/function calling normalization across providers
- Structured output: JSON mode, JSON schema, Pydantic parsing, Instructor, retries on validation failure
- Token counting per provider (tiktoken, Anthropic count_tokens)
- Prompt caching (Anthropic cache_control, OpenAI automatic caching) & cache-aware prompt layout
- Cost tracking per request/user/feature; budgets & quotas
- API key management, secrets, key rotation, per-tenant keys
- Extended thinking / reasoning effort parameters
- Multimodal inputs via API (images, PDFs, audio, files API)
- Embedding API adapters (OpenAI, Cohere, Voyage, Jina, local)
- Response caching (exact-match) at the gateway
- Request/response logging & redaction
- Deterministic testing: mock LLM clients, record/replay (VCR)

---

## 3. PROMPT ENGINEERING LAYER
- System prompts, role prompting, persona
- Zero-shot, one-shot, few-shot; example selection (static, dynamic/semantic)
- Chain-of-thought, zero-shot CoT, step-back prompting, least-to-most
- Self-consistency, tree-of-thought, graph-of-thought, ReAct prompting
- Reflexion / self-critique / self-refine
- Instruction hierarchy; delimiters; XML tags; markdown structuring
- Output formatting: JSON, tables, schema instructions, "think then answer"
- Prompt templates (Jinja2, LangChain PromptTemplate, f-strings), versioning
- Prompt chaining, decomposition
- Meta-prompting, prompt generation, automatic prompt optimization (DSPy, OPRO, APE, TextGrad)
- DSPy programs: signatures, modules, optimizers (BootstrapFewShot, MIPRO)
- Context engineering: what goes in context, ordering, "lost in the middle", context budgeting
- Prompt compression (LLMLingua), summarization of history
- Prompt caching-friendly structure (static prefix, dynamic suffix)
- Prompt registry & management (Langfuse prompts, PromptLayer, Humanloop)
- Prompt injection awareness & defensive prompting
- Multilingual prompting, translation chains
- Prompt evaluation loops (A/B, regression suites)
- Long-context prompting vs RAG decision

---

## 4. DATA & INGESTION LAYER (pre-RAG data pipeline)
### 4.1 Sources & connectors
- Files: PDF, DOCX, PPTX, XLSX, CSV, HTML, Markdown, JSON, TXT, EPUB
- Loaders: LangChain document loaders, LlamaIndex readers, Unstructured, Docling, LlamaParse, PyMuPDF, pdfplumber, Marker
- Web scraping & crawling (BeautifulSoup, Playwright, Scrapy, Firecrawl, Crawl4AI, Jina Reader)
- APIs & SaaS connectors: Google Drive, Notion, Confluence, SharePoint, Slack, Jira, GitHub, S3, databases
- Databases: SQL, NoSQL, data warehouses; CDC / incremental sync
- Email, transcripts, audio→text (Whisper), video
- OCR (Tesseract, PaddleOCR, Azure Document Intelligence, AWS Textract), vision-LLM parsing
- Table extraction, layout-aware parsing, multi-column PDFs, scanned docs
### 4.2 Cleaning & normalization
- Text cleaning (boilerplate removal, header/footer stripping, unicode normalization)
- Language detection
- **Deduplication**: exact hash, near-duplicate (MinHash/SimHash/LSH), semantic dedup (embedding cosine threshold)
- PII detection & redaction (Presidio, regex, NER)
- Content filtering / quality scoring
- Document versioning & change detection (content hashing)
- Metadata extraction (title, author, date, source, section, page, URL, ACLs)
- Schema normalization into a canonical `Document` object
### 4.3 Chunking
- Fixed-size, token-based, sentence-based, paragraph-based
- Recursive character splitting
- Semantic chunking (embedding-based breakpoints)
- Document-structure-aware (Markdown headers, HTML, code AST, tables)
- Hierarchical / parent-child chunking (small-to-big)
- Sliding window with overlap
- Late chunking (Jina), contextual chunk headers, contextual retrieval (Anthropic: LLM-generated chunk context)
- Proposition / sentence-level chunking
- Agentic chunking (LLM-decided boundaries)
- Chunk size tuning & evaluation
### 4.4 Enrichment
- LLM-generated summaries, titles, keywords, questions (HyDE-style doc-to-question)
- Entity/relationship extraction (for graph RAG)
- Classification/tagging, taxonomy
- Multi-representation indexing (summary index + raw index)
### 4.5 Embeddings
- Dense embedding models: OpenAI text-embedding-3, Cohere embed v3/v4, Voyage, Jina, BGE, E5, GTE, Nomic, Qwen-embedding, sentence-transformers
- Multilingual embeddings, code embeddings, long-context embeddings
- Matryoshka embeddings (dimension truncation), binary/int8 quantized embeddings
- Sparse embeddings (SPLADE, BM25 vectors), learned sparse
- Multi-vector / late interaction (ColBERT, ColPali for documents-as-images)
- Multimodal embeddings (CLIP, SigLIP, ImageBind, Cohere multimodal)
- Embedding fine-tuning (contrastive, hard negatives, synthetic pairs)
- Batch embedding, caching embeddings, embedding versioning & re-indexing strategy
- Choosing embedding dims/cost tradeoffs; MTEB benchmark
### 4.6 Indexing & storage
- Vector stores: FAISS, Chroma, Qdrant, Weaviate, Milvus, Pinecone, pgvector, LanceDB, Elasticsearch/OpenSearch, Redis, MongoDB Atlas Vector, Azure AI Search, Vertex Vector Search, Turbopuffer
- Index types: flat, IVF, HNSW, PQ/SQ compression, DiskANN
- Distance metrics (cosine, dot, L2)
- Metadata filtering, payload indexing, pre- vs post-filtering
- Namespaces / multi-tenancy / collections per tenant
- Upsert, delete, re-index, versioned indexes (blue/green index swap)
- Hybrid indexes (dense + sparse in one store)
- Document store + vector store separation (docstore, KV store)
- Incremental indexing pipelines, scheduled sync, idempotent ingestion
- Ingestion orchestration (Airflow, Prefect, Dagster, LlamaIndex IngestionPipeline)
- Ingestion observability (counts, failures, latency, dedup rates)

---

## 5. RAG — RETRIEVAL-AUGMENTED GENERATION (full layer stack)
### 5.1 Basic RAG
- Naive RAG: embed → retrieve top-k → stuff prompt → generate
- RAG from scratch (no framework) with numpy + API
- RAG with LangChain, LlamaIndex, Haystack, LangGraph
- Citations / source attribution in answers
- Chat with PDF, chat with website, chat with codebase
### 5.2 Query layer (pre-retrieval)
- Query understanding & intent classification
- Query rewriting / reformulation
- Query expansion, multi-query generation
- HyDE (hypothetical document embeddings)
- Step-back queries
- Query decomposition (sub-questions), multi-hop planning
- Query routing (which index/collection/tool/datasource)
- Conversational query condensation (chat history → standalone question)
- Spelling correction, entity linking
- Metadata / filter extraction from natural language (self-query retriever)
- Text-to-SQL, text-to-Cypher, text-to-API as retrieval
- Query embedding caching
### 5.3 Retrieval layer
- Dense (vector) retrieval, top-k, similarity threshold
- Sparse / keyword retrieval (BM25, TF-IDF, Elasticsearch)
- Hybrid retrieval + fusion (Reciprocal Rank Fusion, weighted scores)
- Metadata filtering, time-aware/recency-weighted retrieval
- MMR (maximal marginal relevance) for diversity
- Parent-document / small-to-big retrieval, auto-merging retrieval
- Sentence-window retrieval
- Multi-index / multi-collection retrieval, ensemble retrievers
- Recursive retrieval, hierarchical retrieval (summary → chunk)
- Multi-hop / iterative retrieval
- Multi-vector retrieval (ColBERT, ColPali)
- Knowledge graph retrieval (GraphRAG, Neo4j, entity-centric, community summaries)
- SQL/structured retrieval, tabular RAG
- Code retrieval (AST chunking, symbol search)
- Multimodal retrieval (images, tables, charts, video frames)
- Web search retrieval (Tavily, Serper, Exa, Brave, Perplexity API) — "web RAG"
- Long-context retrieval vs full-document stuffing
- Retrieval latency optimization, ANN tuning (ef_search, nprobe)
- Access control / permission-aware retrieval (ACL filtering, row-level security)
### 5.4 Post-retrieval layer
- Reranking: cross-encoders (bge-reranker, ms-marco), Cohere Rerank, Voyage rerank, Jina rerank, LLM-as-reranker, ColBERT rerank
- **Result deduplication** (same chunk from multiple retrievers, near-duplicate chunks, overlap merging)
- Relevance filtering / score thresholding
- Context compression (LLMLingua, extractive compression, LLM summarization of chunks)
- Context window packing & ordering (relevant-first, "lost in the middle" mitigation)
- Token budgeting for context
- Chunk merging / neighboring-chunk stitching
- Metadata injection into context (source, date, section headers)
- Lost-context guards: "I don't know" when retrieval is weak
### 5.5 Generation layer
- Prompt templates for grounded answering, refusal when unsupported
- Citation formatting & inline references, quote extraction
- Structured RAG output (JSON answers with sources)
- Streaming RAG answers with progressive citations
- Answer post-processing: hallucination check, faithfulness check, citation verification
- Follow-up question generation
### 5.6 Advanced RAG architectures
- Self-RAG (reflection tokens / self-grading)
- Corrective RAG (CRAG) — grade docs, fall back to web search
- Adaptive RAG (route: no-retrieval / single / multi-step)
- Fusion RAG (RAG-Fusion)
- Iterative / FLARE (active retrieval)
- RAPTOR (recursive tree summaries)
- GraphRAG (Microsoft), LightRAG, HippoRAG, knowledge-graph-enhanced RAG
- Contextual retrieval (Anthropic), Late chunking
- Multimodal RAG (ColPali/vision-LLM over page images; image+text)
- Agentic RAG (see section 9)
- Long-context RAG & hybrid long-context/retrieval
- Cache-augmented generation (CAG) with KV-cache preloading
- RAG over structured + unstructured (SQL + vectors)
- Temporal RAG, versioned knowledge
- Multi-tenant RAG (per-tenant isolation)
- Federated / multi-source RAG
- Conversational RAG with memory
- RAG for code (repo Q&A, code generation with retrieval)
- Fine-tuning for RAG (RAFT), retrieval-aware fine-tuning
### 5.7 RAG evaluation
- Retrieval metrics: recall@k, precision@k, MRR, NDCG, hit rate
- Generation metrics: faithfulness, answer relevance, context precision/recall (RAGAS), groundedness
- Golden dataset creation, synthetic Q&A generation from docs
- LLM-as-judge for RAG, pairwise comparisons
- Tools: RAGAS, TruLens, DeepEval, LangSmith evals, Arize Phoenix, Giskard
- Ablation: chunk size, k, embedding model, reranker, hybrid weights
- Online evaluation: thumbs up/down, implicit signals
### 5.8 RAG production layers
- Ingestion pipeline (see 4) with dedup, versioning, incremental updates
- Index refresh & re-embedding on model upgrade; dual-write migrations
- Semantic caching of queries/answers (GPTCache, Redis semantic cache)
- Retrieval logging, tracing per stage (query → retrieve → rerank → generate)
- Guardrails on inputs/outputs (see 13)
- Latency budgets per stage, async parallel retrieval
- Cost control (rerank only top-N, cheap model routing)
- Feedback loops: user ratings → eval dataset → prompt/retriever tuning
- Multi-tenancy, RBAC, document-level permissions
- Freshness SLAs, stale data detection
- RAG API service (FastAPI) with streaming, auth, rate limiting

---

## 6. MEMORY LAYER
- Conversation buffer memory, window memory, token-limited memory
- Summary memory (rolling summarization), summary-buffer hybrid
- Entity memory, knowledge-graph memory
- Vector-store long-term memory (semantic recall of past turns)
- Episodic vs semantic vs procedural memory
- Working memory / scratchpad for agents
- Memory write policies (what to store), memory consolidation, forgetting/decay
- User profile memory & personalization
- Cross-session memory, memory per user/tenant
- Memory frameworks: Mem0, Zep, Letta (MemGPT), LangGraph checkpointers & stores, LlamaIndex memory
- Persistence: Redis, Postgres, SQLite checkpointing
- Memory privacy, PII in memory, deletion/right-to-be-forgotten
- Context window management: truncation, compaction, summarization on overflow
- Prompt-cache-friendly memory layout

---

## 7. TOOLS, PROTOCOLS & SKILLS LAYER
### 7.1 Tool / function calling
- Function calling with OpenAI, Anthropic, Gemini; tool schemas (JSON schema)
- Tool definition from Pydantic / type hints / docstrings
- Parallel tool calls, sequential tool loops, tool choice forcing
- Tool result handling, error returns to the model, retries
- Tool registries, dynamic tool selection (RAG over tools)
- Built-in tools: web search, code execution/interpreter, file search, computer use, bash, text editor
- Tool sandboxing (Docker, E2B, Modal, Daytona), timeouts, resource limits
- Human-in-the-loop approval for sensitive tools
- Tool-call validation, argument guardrails
- Idempotency & side-effect safety
### 7.2 Model Context Protocol (MCP)
- MCP concepts: servers, clients, hosts; tools, resources, prompts, sampling, roots, elicitation
- Build an MCP server (Python FastMCP / TypeScript SDK) exposing tools & resources
- Build an MCP client; connect to Claude Desktop / Claude Code / Cursor
- Transports: stdio, Streamable HTTP, SSE (legacy); auth (OAuth 2.1)
- Remote MCP servers, MCP gateways, MCP registries
- MCP security (tool poisoning, prompt injection via tool descriptions), permissions
- Testing MCP servers (inspector)
### 7.3 Agent Skills & agent-facing docs
- Agent Skills (SKILL.md: frontmatter, progressive disclosure, scripts/resources)
- Writing skills for Claude Code / Claude apps; skill triggering & evaluation
- Slash commands, custom instructions (CLAUDE.md, .cursorrules, AGENTS.md)
- Hooks (pre/post tool hooks), subagent definitions
- Plugins / marketplaces for agent tools
### 7.4 Inter-agent & other protocols
- A2A (Agent-to-Agent protocol), agent cards, task lifecycle
- AG-UI (agent ↔ UI event protocol)
- ACP, OpenAI Agents SDK handoffs, LangGraph Platform / Agent Protocol
- OpenAPI → tools, GraphQL tools, database tools (text-to-SQL tools)
- Computer use / browser automation tools (Playwright MCP, Browser Use, Anthropic computer use)

---

## 8. AGENTS LAYER (single-agent design)
### 8.1 Core concepts
- What is an agent: LLM + tools + loop + state; agent vs workflow
- Agentic loop from scratch (while loop: think → act → observe)
- ReAct pattern; plan-and-execute; reflection; ReWOO
- Agent state, scratchpad, message history management
- Termination conditions, max iterations, stuck detection
- Streaming agent events (tokens, tool calls, intermediate steps)
### 8.2 Workflow patterns (Anthropic "Building effective agents")
- Prompt chaining
- Routing (classifier → specialized handler)
- Parallelization (sectioning, voting)
- Orchestrator–workers
- Evaluator–optimizer (generate → critique → refine)
- Autonomous agent loop
### 8.3 Planning & reasoning
- Task decomposition, plan generation & re-planning
- Tree/graph search over actions (LATS, MCTS)
- Reflection, self-critique, Reflexion memory
- Extended thinking / reasoning models in agents
- Goal tracking, todo lists, task queues
### 8.4 Agent frameworks (build the same agent in each)
- Raw API (no framework) — must do first
- LangGraph (StateGraph, nodes/edges, conditional edges, checkpointers, interrupts, human-in-the-loop, subgraphs, streaming, persistence, time travel)
- LangChain agents (create_agent / tool-calling agent)
- LlamaIndex Workflows & agents (FunctionAgent, ReActAgent, AgentWorkflow)
- OpenAI Agents SDK (agents, handoffs, guardrails, tracing, sessions)
- Anthropic Claude Agent SDK (Claude Code as a library: tools, hooks, subagents, MCP, permissions)
- Anthropic Managed Agents (server-hosted agents with sandbox)
- Google ADK (Agent Development Kit), Gemini agents
- Microsoft Agent Framework (AutoGen + Semantic Kernel merger), AutoGen
- CrewAI (agents, tasks, crews, flows, processes, memory, tools)
- Pydantic AI (typed agents, dependencies, result validation)
- smolagents (Hugging Face, CodeAgent)
- Haystack pipelines/agents, DSPy agents (ReAct module)
- Agno (Phidata), Mastra (TS), Vercel AI SDK agents (TS), LangChain.js
- Semantic Kernel, Spring AI (Java), Bedrock Agents / AgentCore, Vertex Agent Builder, Azure AI Foundry agents
- Low-code: Dify, Flowise, Langflow, n8n
### 8.5 Agent types to build
- Conversational assistant with tools
- Research agent (search → read → synthesize with citations)
- Coding agent (read/edit files, run tests, git) — Claude Code-like
- Data analysis agent (pandas + code execution)
- SQL agent (schema discovery → query → verify)
- Browser / computer-use agent
- Customer support agent (ticketing, KB, escalation)
- Email / calendar assistant with approvals
- Workflow automation agent (n8n/Zapier-like)
- Deep research agent (multi-step, long-running)
- Voice agent (STT → LLM → TTS, realtime APIs)
- Document processing agent (extraction pipeline)
### 8.6 Agent state, persistence & control
- Checkpointing, resumable runs, durable execution (Temporal, Inngest, LangGraph Platform, Restate)
- Human-in-the-loop: interrupts, approvals, edits, breakpoints
- Long-running/background agents, async job queues (Celery, RQ, Arq)
- Time travel / replay from checkpoint
- Session management, threads, multi-user isolation
- Agent configuration & versioning
### 8.7 Agent safety & reliability
- Permission systems (allow/deny lists per tool), sandboxes
- Prompt injection via tool results & web content; instruction/data boundary
- Output validation, schema enforcement
- Loop/cost limits, budget guards, kill switches
- Audit logs of every action
- Least privilege credentials for tools
- Testing agents: unit tests for tools, scenario tests, simulation, trajectory evaluation

---

## 9. AGENTIC RAG (agents + retrieval combined)
- Retrieval as a tool (agent decides when/whether to retrieve)
- Router agent → multiple indexes/tools (vector, SQL, web, API)
- Query planning agent: decompose → parallel retrieve → synthesize
- Multi-hop agentic retrieval with reflection ("do I have enough?")
- Corrective RAG as a LangGraph graph (grade → rewrite → web fallback)
- Self-RAG / Adaptive RAG as graphs
- Agentic document Q&A over multiple docs (per-doc sub-agents, doc summaries as index)
- Tool-augmented RAG (calculator, code exec on retrieved tables)
- Deep research agents (iterative search + retrieval + report writing)
- Knowledge-graph agents (Cypher generation + vector hybrid)
- Memory + RAG (personal knowledge base per user)
- Agentic ingestion (agent parses/enriches/verifies documents)
- Citation verification agent, hallucination-check loops
- Evaluation of agentic RAG (trajectory + answer quality, tool-use correctness)
- Latency/cost tradeoffs of agentic vs static RAG; when to use each
- Streaming intermediate steps to the UI (retrieved sources shown live)

---

## 10. MULTI-AGENT SYSTEMS
- Why multi-agent: context isolation, specialization, parallelism
- Topologies: supervisor/orchestrator, hierarchical, peer-to-peer/swarm, pipeline, network, debate, round-robin
- Handoffs (OpenAI Agents SDK, Swarm), delegation, sub-agents (Claude Code subagents)
- Shared state vs message passing; blackboard pattern
- Orchestrator–worker with parallel workers (research fan-out / fan-in)
- Multi-agent frameworks: CrewAI (crews & flows), AutoGen / Microsoft Agent Framework (group chat, selector), LangGraph multi-agent (supervisor, hierarchical teams, swarm), OpenAI Agents SDK, Google ADK (sequential/parallel/loop agents), MetaGPT, CAMEL, ChatDev, Agno teams, LlamaIndex AgentWorkflow
- Role design, agent personas, task specs, tool partitioning
- Inter-agent communication protocols (A2A), agent registries/discovery
- Consensus, voting, debate, critic agents, judge agents
- Conflict resolution, deadlock/loop prevention, budget across agents
- Observability across agents (distributed tracing), cost attribution per agent
- Evaluation of multi-agent systems, when NOT to use multi-agent
- Multi-agent with RAG (each specialist has its own index)
- Human as an agent in the loop

---

## 11. MULTIMODAL GENAI
### 11.1 Vision
- Vision-language models (GPT-4o/5, Claude, Gemini, Qwen-VL, LLaVA, Pixtral, InternVL)
- Image understanding, OCR via VLM, chart/table/diagram reading, document VQA
- Image generation: diffusion (Stable Diffusion, FLUX, SDXL), DALL·E, Imagen, Midjourney API-likes; ControlNet, LoRA for diffusion, inpainting/outpainting
- Image editing models, image-to-image
- Vision embeddings (CLIP/SigLIP), image search, ColPali doc retrieval
- Video understanding (Gemini, frame sampling), video generation (Sora, Veo, Runway, Kling)
- Object detection / segmentation with foundation models (SAM, Grounding DINO, OWL-ViT)
### 11.2 Audio & speech
- STT: Whisper, Deepgram, AssemblyAI, Gemini audio; diarization
- TTS: ElevenLabs, OpenAI TTS, Cartesia, Kokoro, Coqui
- Realtime voice agents (OpenAI Realtime API, Gemini Live, LiveKit, Pipecat, Vapi)
- Voice activity detection, interruption handling, latency budgets
- Music/sound generation (Suno-like, AudioCraft)
- Audio embeddings, audio RAG (podcast search)
### 11.3 Multimodal apps
- Multimodal RAG (text+image+table), PDF-as-images RAG
- Multimodal agents (screenshots → actions; computer use)
- Document understanding pipelines (invoice/receipt/form extraction)
- Image captioning, alt-text generation, visual moderation

---

## 12. EVALUATION & OBSERVABILITY LAYER
### 12.1 Evaluation
- Why evals; offline vs online; eval-driven development
- Golden datasets, synthetic test generation, human labeling
- Metrics: exact match, F1, BLEU/ROUGE (legacy), BERTScore, semantic similarity
- LLM-as-judge (single, pairwise, rubric-based), judge calibration & bias, G-Eval
- RAG metrics (RAGAS: faithfulness, context precision/recall, answer relevancy)
- Agent evals: task success, trajectory eval, tool-call correctness, step efficiency, cost
- Structured output validity rate, schema compliance
- Safety evals: toxicity, bias, jailbreak resistance, PII leakage
- Benchmarks: MMLU, GPQA, HumanEval, SWE-bench, GAIA, τ-bench, BFCL (tool calling), MTEB (embeddings), BEIR (retrieval), LiveBench, Arena
- Regression testing prompts in CI (pytest + eval frameworks)
- Frameworks: DeepEval, RAGAS, promptfoo, Inspect (UK AISI), OpenAI Evals, LangSmith, Braintrust, Arize Phoenix, TruLens, Giskard, Weights & Biases Weave, MLflow LLM eval
- A/B testing prompts/models in production, interleaving
- Statistical significance, variance across runs, temperature=0 vs sampling in evals
- Red teaming (manual + automated: PyRIT, Garak, Promptfoo red team)
### 12.2 Observability / tracing
- OpenTelemetry for LLM apps; GenAI semantic conventions
- Tracing: spans per LLM call/tool/retrieval; LangSmith, Langfuse, Arize Phoenix, Helicone, Braintrust, OpenLLMetry, Datadog LLM Observability, W&B Weave, Opik
- Logging prompts/completions with redaction; sampling
- Metrics: latency (p50/p95), TTFT, tokens in/out, cost, error rates, tool failure rates, cache hit rates
- Dashboards & alerts (Grafana, Prometheus)
- User feedback capture & linking to traces
- Dataset curation from production traces (traces → evals)
- Drift detection (input distribution, output quality over time)

---

## 13. SAFETY, SECURITY & GUARDRAILS LAYER
- Threat model: prompt injection (direct/indirect), jailbreaks, data exfiltration, tool abuse, excessive agency, supply-chain (poisoned models/MCP servers) — OWASP Top 10 for LLM Apps
- Input guardrails: prompt-injection classifiers (Lakera, Prompt Guard, Llama Guard), topic restriction, PII detection/redaction, toxicity, language checks
- Output guardrails: hallucination/groundedness checks, PII leakage, toxicity, format validation, policy compliance, secret detection
- Guardrail frameworks: NeMo Guardrails, Guardrails AI, LLM Guard, Llama Guard, OpenAI Moderation, Azure Content Safety, Bedrock Guardrails
- Constitutional / policy prompts, system-prompt hardening, instruction/data separation (delimiters, spotlighting)
- Tool permissioning, sandboxing, least privilege, approval flows
- Data privacy: on-prem/local models, zero-retention APIs, encryption, data residency, GDPR deletion
- Secrets management, API key scoping, per-user auth passthrough to tools
- Rate limiting & abuse prevention, cost attack mitigation (token bombs)
- Content watermarking, provenance (C2PA)
- Bias & fairness testing, explainability
- Compliance: EU AI Act, NIST AI RMF, ISO 42001, model cards, audit trails
- Model security: adversarial inputs, model extraction, membership inference (awareness)
- Red teaming programs & incident response for AI features

---

## 14. PRODUCTION PLATFORM LAYER (cross-cutting infrastructure)
- **Adapter layer** (provider abstraction; see 2)
- **Gateway / router** (model routing, fallbacks, retries, circuit breakers)
- **Caching**: exact-match response cache, semantic cache, embedding cache, prompt cache (provider-side KV), retrieval cache; cache invalidation
- **Deduplication**: ingestion dedup (docs/chunks), retrieval result dedup, request dedup (idempotency keys), duplicate tool-call suppression, duplicate-message detection in chat, near-duplicate output detection
- **Rate limiting & quotas** per user/tenant/key; queueing & backpressure
- **Batching**: batch APIs, micro-batching embeddings, batch inference jobs
- **Async & streaming**: SSE/WebSocket streaming, async pipelines, partial results, cancellation
- **Concurrency control**, connection pooling, worker pools
- **Cost management**: token accounting, per-feature budgets, model tiering (cheap→expensive cascade), prompt trimming, output length limits, FinOps dashboards
- **Latency optimization**: parallel calls, speculative prefetch of retrieval, smaller models for sub-tasks, prompt caching, streaming UX
- **Reliability**: retries/backoff/jitter, timeouts, fallbacks, graceful degradation, dead-letter queues
- **Configuration & feature flags** for prompts/models; prompt & model versioning; rollout/rollback
- **Multi-tenancy**: tenant isolation of data, indexes, keys, budgets
- **AuthN/AuthZ**: OAuth/JWT, RBAC/ABAC, document-level ACL propagation into retrieval
- **Data layer**: Postgres (+pgvector), Redis, object storage, message queues (Kafka, SQS, RabbitMQ)
- **Job orchestration**: Celery/Arq/Temporal for long-running agent runs; scheduled re-indexing
- **Observability** (see 12.2), audit logging, PII redaction in logs
- **Testing**: unit (tools, parsers), contract tests for adapters, mock LLM, snapshot/golden tests, eval suites in CI, load testing (Locust/k6) with streaming
- **API design**: FastAPI/Flask/Node service, OpenAPI, versioned endpoints, streaming endpoints, webhooks/callbacks for async agents
- **Frontend integration**: chat UI (Streamlit, Gradio, Chainlit, Next.js + Vercel AI SDK, assistant-ui, CopilotKit/AG-UI), rendering tool calls & citations, optimistic streaming, generative UI
- **Human-in-the-loop UX**: approvals, edits, feedback widgets
- **Compliance & governance**: data retention policies, model inventory, usage policies
- **Documentation & runbooks** for AI features; on-call for LLM incidents

---

## 15. DEPLOYMENT, MLOPS / LLMOPS & INFRA
- Containerizing LLM apps (Docker), docker-compose with vector DB + app + Redis
- Kubernetes deployment, HPA, GPU node pools, KServe/Ray Serve/Triton
- Serverless (AWS Lambda, Cloud Run, Modal, Vercel) for LLM APIs; cold starts & streaming
- Cloud AI platforms: AWS Bedrock (+ Knowledge Bases, Agents, AgentCore, Guardrails), Azure OpenAI/AI Foundry/AI Search, GCP Vertex AI/Gemini/Agent Builder
- Self-hosting open models (vLLM on GPU, Ollama on-prem), GPU sizing (VRAM math), spot instances
- CI/CD for prompts, evals as gates, model/prompt registries
- Model registry & versioning (MLflow, Hugging Face Hub), artifact storage
- Experiment tracking (W&B, MLflow), hyperparameter sweeps for fine-tuning
- Blue/green & canary releases for model swaps; shadow traffic
- Secrets (Vault, AWS Secrets Manager), IAM
- Edge/on-device inference (llama.cpp, MLX, ONNX, CoreML, WebLLM, WebGPU, Android AICore)
- Monitoring GPU utilization, autoscaling on queue depth
- Disaster recovery for indexes; backup/restore of vector DBs
- Data pipelines infra (Airflow/Prefect/Dagster) for ingestion & re-embedding
- Infrastructure as code (Terraform) for AI stacks

---

## 16. APPLICATION PATTERNS (end-to-end projects to build)
- Chatbot with memory + streaming UI
- Enterprise RAG knowledge assistant (multi-source, ACLs, citations, evals, observability)
- Document extraction pipeline (PDF → structured JSON → DB) with validation
- Text-to-SQL analytics assistant with chart generation
- Customer support agent with ticketing tools + escalation + guardrails
- Coding assistant / code review bot (GitHub webhooks + agent)
- Deep research report generator (multi-agent + web + RAG)
- Meeting assistant (audio → transcript → summary → action items → calendar)
- Voice agent (realtime)
- Content generation pipeline (brief → draft → critique → SEO → publish) with human approval
- Recommendation / personalization with embeddings
- Semantic search engine (hybrid + rerank) over a product catalog
- Classification/extraction at scale with batch APIs and dedup
- Email triage & auto-reply agent with approvals
- Browser automation agent (form filling, scraping with vision)
- Internal copilot over Slack/Confluence/Jira via MCP servers
- Multimodal app: image-based product search or invoice processing
- Fine-tuned domain SLM deployed locally + eval vs API model
- Multi-tenant SaaS AI feature with cost tracking & quotas
- AI evaluation harness + dashboard for any of the above

---

## 17. ADVANCED / EMERGING (2025–2026)
- Reasoning models & test-time compute; controlling thinking budgets
- Agentic coding at scale (Claude Code, Codex, Cursor agents, background agents, parallel worktrees)
- Computer-use agents & browser agents in production
- Long-running autonomous agents, durable execution, agent orchestration platforms
- Managed agent runtimes (Anthropic Managed Agents, Bedrock AgentCore, OpenAI Responses API + hosted tools, LangGraph Platform)
- MCP ecosystem: remote servers, auth, registries, gateway security
- A2A / multi-vendor agent interoperability
- Agent Skills & progressive-disclosure knowledge for agents
- Context engineering as a discipline (compaction, memory files, sub-agent context isolation)
- Small models on-device; hybrid local+cloud routing
- Diffusion LLMs, mixture-of-depths, linear attention, 1M+ context models
- Synthetic data flywheels, self-improving systems, RL for agents (RLVR, agentic RL environments)
- World models, embodied/robotics VLA models (awareness)
- Generative UI, AI-native product design
- AI FinOps, token economics, cost-aware architectures
- Governance: AI Act compliance tooling, model provenance, evals-as-contracts

---

### Suggested practice order
0 → 2 → 3 → 4 → 5 (naive→advanced) → 12 (evals early!) → 6 → 7 → 8 → 9 → 10 → 13 → 14 → 15 → 11 → 1 (training/inference deep-dive) → 16 (capstones) → 17

---

# ADDENDUM — gaps found on re-audit (slotted into their layers)

## → Slot into 3 (Prompting): CORE LLM TASKS (practice each as a program before RAG/agents)
- Summarization: single-doc, long-doc (map-reduce, refine, chain-of-density), meeting/transcript summaries
- Classification & sentiment (zero-shot vs few-shot vs fine-tuned encoder; cost/accuracy tradeoff)
- Information extraction → structured JSON (entities, relations, key-value, tables)
- Translation & localization, style transfer, rewriting, grammar correction
- Q&A (closed-book vs open-book), reading comprehension
- Text generation (marketing copy, emails), controlled generation (length/tone/format)
- Code generation, code explanation, fill-in-the-middle (FIM), code repair, test generation
- Data labeling / annotation with LLMs (plus human review: Label Studio, Argilla)
- Topic modeling & clustering with embeddings (BERTopic, k-means on embeddings)
- Semantic similarity / paraphrase detection / duplicate detection
- Keyword & question generation, title generation
- Uncertainty & calibration: logprob-based confidence, abstention ("I don't know"), verbalized confidence
- Hallucination detection methods: SelfCheckGPT, NLI-based entailment checks, citation checking

## → Slot into 2 (Model access): PROVIDER-NATIVE FEATURES (as of 2026)
- OpenAI: Responses API (vs Chat Completions), built-in tools (web search, file search, code interpreter, computer use), Batch, Realtime, Codex/agents, structured outputs, reasoning effort
- Anthropic: extended/interleaved thinking, effort parameter, prompt caching, Message Batches, Citations API, PDF/Files API, structured outputs, server tools (web search/fetch, code execution, computer use, text editor, bash), tool search, memory tool, context editing/compaction, Claude Agent SDK, Managed Agents, Claude Code (hooks, subagents, skills, plugins, MCP)
- Google: Gemini context caching, Grounding with Google Search, File Search / Vertex RAG Engine, Live API, ADK, Veo/Imagen APIs
- AWS Bedrock: Converse API, Knowledge Bases, Agents/AgentCore, Guardrails, model import
- Azure: AI Foundry, Azure AI Search integrated vectorization, Azure OpenAI On Your Data
- Provider-managed RAG (OpenAI File Search, Gemini File Search, Bedrock KB) vs self-built RAG — when to use which
- Open-weight model landscape & licenses: Llama, Qwen, DeepSeek, Mistral/Devstral, Gemma, Kimi K2, GLM, Phi, SmolLM; license implications

## → Slot into 1 (Model layer): missing training/inference items
- Tokenizer training (train your own BPE with `tokenizers`), vocab extension for new languages
- Hugging Face ecosystem: `datasets` (streaming, map, dedup), Hub, `accelerate`, Spaces, `evaluate`
- Data versioning (DVC, LakeFS), dataset cards
- Training infra on cloud: SageMaker, Vertex Training, Lambda Labs, RunPod, multi-node setup
- lm-evaluation-harness for base/fine-tuned model benchmarking
- Inference optimizations: torch.compile, CUDA graphs, ONNX Runtime, OpenVINO, TensorRT, kernel fusion
- Long-context inference: chunked prefill, ring attention, KV-cache offloading/compression
- Model merging & upcycling to MoE; continual learning
- Mechanistic interpretability (awareness): sparse autoencoders, activation steering/steering vectors, probing, logit lens
- Model licensing, model cards, responsible release

## → Slot into 4/5 (Data & RAG): missing retrieval items
- Provider-native embeddings dimension selection & re-indexing migration playbook
- Retrieval for tabular data (row/column embeddings, schema linking), spreadsheet RAG
- Chunk-level metadata for citations (page numbers, bounding boxes, char offsets)
- Anthropic Citations API / Gemini grounding metadata for verifiable answers
- Document-image retrieval (ColPali/ColQwen) end-to-end pipeline
- Query classification to skip retrieval (chit-chat vs knowledge question)
- Retrieval unit tests & golden retrieval sets in CI

## → Slot into 8/10 (Agents): missing agent items
- Simulated-user testing for agents (τ-bench style), agent sandboxes/environments (gym-like)
- RL fine-tuning for agents on tool-use trajectories (agentic RL, RLVR environments)
- Cost/latency-aware agent design: model cascades inside agents (cheap planner, strong executor)
- Agent observability standards (OpenTelemetry GenAI/agent spans), replay debugging
- Parallel agent execution in isolated worktrees / containers (background coding agents)
- Agent-to-UI streaming protocols (AG-UI, Vercel AI SDK data streams, generative UI)
- Multi-agent failure taxonomy (MAST) & mitigations

## → Slot into 11 (Multimodal): missing
- 3D generation (Meshy, Shap-E, Gaussian splatting — awareness), image-to-3D
- Speech-to-speech models, voice cloning ethics/consent
- Video RAG (frame/scene embeddings), audio RAG

## → Slot into 12/13 (Eval & Safety): missing
- Automated red-teaming tools (PyRIT, Garak, promptfoo redteam) as CI jobs
- Eval dataset versioning & leakage/contamination checks
- Human eval rubrics, inter-annotator agreement
- Bias/fairness metrics for generated text; toxicity classifiers (Detoxify, Perspective)
- Safety for agents: excessive agency, tool-result injection tests, data-exfiltration tests (markdown image links, URL leaks)

## → Slot into 14/15 (Production): missing
- Notebook → production refactor (Jupyter prototypes to services)
- Cost estimation calculators & token forecasting before launch
- SLA/SLO definition for AI features (quality SLOs, not just latency)
- Prompt/model change management (review process, eval gates, rollback)
- Data retention & zero-data-retention contracts; DPA awareness
- Chaos testing for provider outages; multi-region failover

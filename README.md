# Plan-and-Execute Agent

A **modular AI agent** built around a Plan-and-Execute architecture. The agent decomposes user queries into structured plans, resolves ambiguities through interactive clarification, then executes each step via a ReAct loop backed by a pluggable tool suite — including hybrid RAG retrieval, live web search, datetime utilities, and corpus management.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## �� Why This Project?

This repository demonstrates a full agentic pipeline with every stage explicitly implemented:
- **Clarification Before Planning**: The planner asks targeted follow-up questions when a query is too vague before committing to a plan
- **Structured Planning**: LLM decomposes user intent into an ordered list of self-contained, executable steps with explicit goal and reasoning
- **ReAct Execution Loop**: Each step is executed via a Reasoning + Action loop — the executor dispatches tools, observes results, and decides whether more tool calls are needed
- **Pluggable Tool Suite**: Tools are registered by name and description; the LLM selects the right tool(s) at runtime based purely on their descriptions
- **Hybrid RAG Pipeline**: Dense bi-encoder (FAISS) + BM25 lexical retrieval fused with cross-encoder re-ranking, query rewriting, and query-aware summarization
- **Live Web Search**: Fetches real-time web content, indexes it into a temporary RAG corpus, and returns the top relevant passages
- **Final Synthesis**: After all steps complete, a dedicated compiler combines step results into a coherent, well-structured final answer

It serves both as a learning project and a portfolio demonstration of agentic AI design, advanced RAG, and modular software architecture.

## 🌟 Features

### Agentic Orchestration
- **Interactive Clarification**: Before planning, the planner model asks the user targeted follow-up questions until it has enough context (`needs_clarification=True/False`)
- **Structured Plan Generation**: The planner emits a `PlanningAgentResponse` with a `goal`, `plan_reasoning`, ordered `steps`, and a per-step `max_iterations` budget
- **ReAct Executor Loop**: For each step, the `ExecutorAgent` alternates between tool-calling (Dispatcher) and self-evaluation (Internal Synthesizer) until the step is resolved or the iteration budget is exhausted
- **Final Answer Compilation**: After all steps, a `FINAL_COMPILER` prompt synthesizes all step results into a single coherent answer and optionally suggests a follow-up
- **Conversation History**: `ChatHistoryManager` persists the full conversation to disk and provides a sliding-window view to keep the LLM context bounded.

### Tool Suite
- **`rag_query`** — Full RAG pipeline: query rewriting → hybrid retrieval → context summarization → grounded answer generation with `[N]` inline citations
- **`related_topics_retriever`** — Retrieval-only path: rewrites the query then fetches and returns the top-k raw passages directly, with no summarization or answer generation
- **`web_search`** — Fetches live web pages, indexes them into a temporary corpus, and returns the top relevant raw passages for downstream synthesis
- **`add_data_to_corpus`** — Ingests new documents into a named corpus and re-indexes it
- **`clear_corpus`** — Wipes all documents from a named corpus and resets its index
- **`get_current_datetime`** — Returns the current date and time
- **`get_days_since_epoch`** — Returns the number of days between a target date and Unix epoch
- **`internal_knowledge`** — Calls the model with a `query` parameter and returns an answer using the model's parametric knowledge. Preferred for general knowledge, definitions, science, history, math, reasoning, and coding — anything that does not require live or time-sensitive data.

### Hybrid RAG Pipeline
- Dense bi-encoder (FAISS `IndexFlatIP` with `all-MiniLM-L6-v2`) for fast semantic retrieval
- BM25 (`rank_bm25` + NLTK Porter stemming) for lexical keyword retrieval
- Weighted linear fusion of min-max normalized semantic and lexical scores
- Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) for high-precision final selection
- Sliding-window chunking with configurable length and stride
- Overlapping chunk merging for coherent passage reconstruction
- Session history corpus: past (improved\_query → response) pairs are indexed for cache-based reuse

### Technical Infrastructure
- **Flexible LLM Support**: In the project, Google Gemini (2.5 Flash / 2.5 Pro) is used.
- **Structured LLM Outputs**: Pydantic schemas enforce structured responses for planning, synthesis, and executor steps
- **Shared LLM Logger**: `LLMLogger` writes every prompt/response pair to a timestamped JSON file, tagged by the calling instruction role
- **Modular Architecture**: Clean separation of planning, execution, tool management, RAG, and utilities

## 🏗️ Architecture

```
plan-and-execute-agent/
├── main.py                              # Entry point: wires all components and runs the interactive loop
├── config.py                            # Loads HF_TOKEN and GOOGLE_API_KEY from .env
├── agent/
│   ├── orchestrator.py                  # run_orchestrator(): executes the plan step-by-step and compiles the final answer
│   ├── executor_agent.py                # ExecutorAgent: ReAct loop (Dispatcher → Internal Synthesizer → Final Synthesizer)
│   ├── interaction/
│   │   └── clarification.py             # run_clarification(): interactive clarification loop before planning
│   └── tools/
│       ├── tool_generator.py            # CreateTools: factory methods for all tool instances
│       ├── tool_manager.py              # ToolManager: registry, schema generation (Gemini), and dispatch
│       └── tool_handlers/
│           ├── rag_handler.py           # RAGHandler: full RAG pipeline + corpus management tools
│           ├── web_search_handler.py    # WebSearchHandler: live search → corpus ingestion → retrieval
│           ├── datetime_handler.py      # get_current_datetime(), get_days_since_epoch()
│           └── internal_knowledge_handler.py  # use_internal_knowledge(query, generative_model): calls LLM with DIRECT_RESPONSE prompt
├── rag/
│   ├── rag_corpus_manager.py            # Document chunking, embedding, FAISS & BM25 indexing
│   ├── rag_content_retriever.py         # Hybrid retrieval + cross-encoder re-ranking (corpus-agnostic)
│   └── bm25_keyword_retriever.py        # BM25Okapi wrapper with NLTK tokenization and Porter stemming
├── utils/
│   ├── model_loader.py                  # ModelLoader: Gemini, SentenceTransformer, CrossEncoder, Qwen
│   ├── chat_history_manager.py          # Sliding-window conversation history with JSON persistence
│   ├── llm_logger.py                    # Shared per-run JSON logger for all LLM calls
│   ├── dataset_utils.py                 # DataSetUtil: HuggingFace dataset loading (SQuAD default)
│   ├── web_search.py                    # WebSearch: DuckDuckGo search + Trafilatura page extraction
│   └── logger.py                        # Low-level JSON file logger
└── base/
    ├── constants.py                     # All model names, tool names/descriptions, system prompt templates
    └── data_classes.py                  # All dataclasses and Pydantic schemas (ToolResult, GenerativeModel, Plan*, etc.)
```

## 🚀 Quick Start



### Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_hugging_face_token
GOOGLE_API_KEY=your_google_api_key
```

These are loaded automatically by `config.py` via `python-dotenv`.

### Running the Agent

```bash
python main.py
```

The agent enters an interactive loop. Type your prompt and press Enter. The agent will:
1. Ask follow-up questions if your prompt is ambiguous
2. Generate a structured step-by-step plan
3. Execute each step via the ReAct loop, calling tools as needed
4. Synthesize all step results into a final answer
5. Optionally offer a context-aware follow-up question

---

## 🧠 How It Works

### 1. Clarification & Planning

```
User Prompt
    │
    ▼
Planner LLM (PLANNER system prompt)
    │
    ├─(needs_clarification=True)──► Ask follow-up question → get user answer → loop
    │
    └─(needs_clarification=False)──► PlanningAgentResponse
                                        ├── goal
                                        ├── plan_reasoning
                                        ├── steps: [PlanStep, ...]
                                        └── max_iterations
```

The planner emits a structured `PlanningAgentResponse` (Pydantic model). When `needs_clarification=True`, `run_clarification()` prints the question, reads the user's answer, appends both to `ChatHistoryManager`, and re-queries the planner with the updated history until it is satisfied.

### 2. Plan Execution (Orchestrator + ExecutorAgent)

```
For each step in plan.steps:
    │
    ▼
ExecutorAgent.execute_step()
    │
    ├── [Iteration 1..max_iterations]
    │       │
    │       ├── Dispatcher LLM → selects tool(s) → calls tool(s) via ToolManager
    │       │       └── observation appended
    │       │
    │       └── Internal Synthesizer LLM → evaluates progress
    │               ├─(early_stop=True)──► break
    │               └─(early_stop=False)─► next iteration
    │
    └── Final Synthesizer LLM → produces StepResult.answer
```

After all steps complete, `run_orchestrator()` feeds all step results to the `FINAL_COMPILER` prompt, which synthesizes a single coherent `FinalSynthesisResponse` containing `final_answer` and an optional `follow_up_question`.

### 3. RAG Pipeline (inside `rag_query` tool)

```
Query
  │
  ▼
Query Rewriting (QUERY_REWRITER prompt)
  │
  ▼
History Check (if use_agent_history=True) ──(cache hit)──► Return cached ToolResult
  │ (miss)
  ▼
Hybrid Retrieval
  ├── FAISS bi-encoder  ─┐
  └── BM25 (if weight>0) ├─► min-max normalize → weighted fusion → top-K candidates
                         ┘
  │
  ▼
Cross-Encoder Re-ranking → top-N passages
  │
  ▼
Query-Aware Context Summarization (CONTEXT_SUMMARIZER prompt, per passage)
  │
  ▼
Grounded Generation with Inline Citations [1][2]… (RESPONSE_GENERATOR prompt)
  │
  ▼
Index into History Corpus → return ToolResult(.text, .data)
```

### 4. Web Search Pipeline (inside `web_search` tool)

```
Query
  │
  ▼
Query Optimization (QUERY_OPTIMIZER prompt)
  │
  ▼
DuckDuckGo Search → top URLs
  │
  ▼
HTTP fetch + Trafilatura HTML extraction (per URL)
  │
  ▼
Index extracted pages into web_search corpus (RAGCorpusManager)
  │
  ▼
Retrieval-only RAG (retrieve_top_related_items) → top passages
  │
  ▼
Return ToolResult (raw passages for downstream synthesis)
```

### 5. System Prompt Roles

All prompts are defined in `Constants.Instructions` in `base/constants.py`:

| Constant | Stage | Purpose |
|---|---|---|
| `PlanAndExecuteAgent.PLANNER` | Clarification & Planning | Decompose query into steps; ask for clarification when needed |
| `PlanAndExecuteAgent.Executor.EXECUTOR_DISPATCHER` | ReAct — Act | Select and call the right tool(s) using explicit routing rules: no-tool rule (reasoning/summarization → `internal_knowledge`), internal-knowledge-first (stable facts), temporal-query (live data → `web_search`), corpus-first (warm corpus → RAG before re-searching) |
| `PlanAndExecuteAgent.Executor.EXECUTOR_INTERNAL_SYNTHESIZER` | ReAct — Reflect | Evaluate tool results; set `early_stop=True` when the step is fully answerable, `False` only when a named specific piece of information is still missing — partial answers or vague uncertainty are not reasons to continue |
| `PlanAndExecuteAgent.Executor.EXECUTOR_FINAL_SYNTHESIZER` | Step synthesis | Produce the final answer for one step from all observations |
| `PlanAndExecuteAgent.FINAL_COMPILER` | Final synthesis | Combine all step results into the agent's final answer; when no steps were executed (e.g. user declined a follow-up), respond naturally using the conversation history |
| `WebSearch.QUERY_OPTIMIZER` | Web search | Compress user query into a high-density search engine query |
| `RAG.QUERY_REWRITER` | RAG retrieval | Rewrite query for maximum recall; expand abbreviations and add synonyms |
| `RAG.CONTEXT_SUMMARIZER` | RAG retrieval | Summarize a retrieved passage relative to the query |
| `RAG.RESPONSE_GENERATOR` | RAG generation | Generate a grounded answer with `[N]` inline citations |
| `RAG.CITATION_CHECKER` | RAG verification | Verify every claim against source passages; return `pass/partial/fail` verdict |
| `InternalKnowledge.DIRECT_RESPONSE` | Internal knowledge tool | System prompt for the `internal_knowledge` tool's own generative model call; logs as `caller: "direct_response"` |

## 💡 Usage Examples

### Full Agent Interaction

```python
from agent.executor_agent import ExecutorAgent
from agent.interaction.clarification import run_clarification
from agent.orchestrator import run_orchestrator
from agent.tools.tool_generator import CreateTools
from agent.tools.tool_manager import ToolManager
from base.constants import Constants
from utils.chat_history_manager import ChatHistoryManager
from utils.llm_logger import LLMLogger
from utils.model_loader import ModelLoader
from config import HF_TOKEN, GOOGLE_API_KEY

model_loader = ModelLoader()
llm_logger = LLMLogger(log_dir="./logs")

generative_model = model_loader.load_gemini_generative_model(
    google_api_key=GOOGLE_API_KEY,
    model_name=Constants.ModelNames.Gemini.GEMINI_2_5_FLASH,
    llm_logger=llm_logger,
)

# ... (build corpora, RAGHandler, WebSearchHandler — see main.py for full setup)

tool_manager = ToolManager(Constants.Tools.SupportedLLMs.GEMINI)
# tool_manager.add_tool(...)  # register tools

executor = ExecutorAgent(generative_model=generative_model, tool_manager=tool_manager)
chat_history_manager = ChatHistoryManager(log_dir="./logs", window_size=100)

user_prompt, plan = run_clarification(
    generative_model=generative_model,
    chat_history_manager=chat_history_manager,
)

run_orchestrator(
    user_prompt=user_prompt,
    plan=plan,
    generative_model=generative_model,
    executor=executor,
    chat_history_manager=chat_history_manager,
)
```

### RAG Handler Directly

```python
# Full pipeline: query rewriting → retrieval → summarization → generation
result = rag_handler.get_query_response(
    query="Which NFL team won Super Bowl 50?",
    corpus_name="knowledge_base",
    top_k=3,
)
print(result.text)                      # Grounded answer with citations
print(result.data["response_type"])     # "generative_model" or "agent_history"
print(result.data["context"])           # Summarized passages used for generation

# Retrieval only (no generation)
result = rag_handler.retrieve_top_related_items(
    query="capital of France",
    corpus_name="knowledge_base",
    top_k=5,
)
print(result.text)   # Numbered list of raw passages
print(result.data)   # Raw list of passage strings

# Add custom documents
rag_handler.add_data_to_corpus(
    corpus_name="knowledge_base",
    documents=["Notre Dame was founded in 1842...", "The current president is Rev. John Jenkins..."],
    metadatas=[{"source": "wiki"}, {"source": "official"}],
)

# Clear a corpus
rag_handler.clear_corpus(corpus_name="web_search")
```

### RAG Parameters

Customize chunking in `RAGCorpusManager`:

```python
from rag.rag_corpus_manager import RAGCorpusManager

rag_corpus = RAGCorpusManager(
    sentence_transformer=sentence_transformer,
    max_data_chunk_len=300,   # Maximum words per chunk (default: 300)
    data_chunk_stride=75,     # Overlap between consecutive chunks (default: 75)
)
```

Tune retrieval in `RAGContentRetriever`:

```python
indices, scores = rag_retriever.find_top_similar_items(
    rag_corpus=corpus,
    query=query,
    initial_retrieval_top_k=100,      # Candidates fetched by bi-encoder and BM25 (default: 100)
    top_k=5,                          # Final passages after re-ranking (default: 5)
    cross_encoder_batch_size=16,      # Batch size for cross-encoder inference (default: 16)
    semantic_similarity_weight=0.8,   # Weight for FAISS bi-encoder score (default: 0.8)
    lexical_similarity_weight=0.2,    # Weight for BM25 score (default: 0.2; BM25 is skipped when 0)
)
```

Pass per-step model configs to `get_query_response`:

```python
result = rag_handler.get_query_response(
    query=query,
    corpus_name="knowledge_base",
    top_k=3,
    improve_query_kwargs_dict={"config": config},
    summarize_context_kwargs_dict={"config": config},
    generate_response_kwargs_dict={"config": config},
    rag_retriever_kwargs_dict={"initial_retrieval_top_k": 50},
)
```

Control session history reuse:

```python
# Use cache when cross-encoder score >= 6.5
result = rag_handler.get_query_response(
    query=query,
    corpus_name="knowledge_base",
    similarity_score_threshold=6.5,
    use_agent_history=True,
)

# Skip history check entirely
result = rag_handler.get_query_response(
    query=query,
    corpus_name="knowledge_base",
    use_agent_history=False,
)
```

### Model Selection

Models are configured as constants in `base/constants.py`:

```python
# Embedding & re-ranking (HuggingFace)
Constants.ModelNames.HuggingFace.SENTENCE_EMBEDDING_MINILM_L6_V2        # "all-MiniLM-L6-v2"
Constants.ModelNames.HuggingFace.CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2  # "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generative (Gemini)
Constants.ModelNames.Gemini.GEMINI_2_5_PRO    # "gemini-2.5-pro"
Constants.ModelNames.Gemini.GEMINI_2_5_FLASH  # "gemini-2.5-flash"

# Generative (Qwen / local)
Constants.ModelNames.Qwen.QWEN_2_5_1_5B_INSTRUCT  # "Qwen/Qwen2.5-1.5B-Instruct"
Constants.ModelNames.Qwen.QWEN_2_5_3B_INSTRUCT    # "Qwen/Qwen2.5-3B-Instruct"
```

## 📊 Components

### ExecutorAgent (`agent/executor_agent.py`)
Runs a single plan step via a ReAct (Reason + Act) loop.

**Key method:**
- `execute_step(step_description, max_iterations, context)` — Dispatches tool calls, collects observations, evaluates progress via the internal synthesizer, and produces a final `StepResult` via the final synthesizer.

**Three LLM roles per step:**
- `EXECUTOR_DISPATCHER` — Selects and calls tools based on the step description and current observations, following explicit routing rules: pure reasoning/summarization steps go to `internal_knowledge`; stable general knowledge uses `internal_knowledge` first and escalates only if insufficient; time-sensitive queries go directly to `web_search`; warm corpora are queried via RAG before triggering a new web search
- `EXECUTOR_INTERNAL_SYNTHESIZER` — Audits tool results; sets `early_stop=True` when the step is fully answerable from current observations, `False` only when a concrete specific piece of information is still missing
- `EXECUTOR_FINAL_SYNTHESIZER` — Produces the definitive `StepResult.answer` from all observations

### Orchestrator (`agent/orchestrator.py`)
Sequences step execution and compiles the final answer.

**Key function:**
- `run_orchestrator(user_prompt, plan, generative_model, executor, chat_history_manager)` — Iterates over `plan.steps`, feeds prior step results as context, and calls the `FINAL_COMPILER` prompt to produce a `FinalSynthesisResponse`. When `plan.steps` is empty (e.g. user declined a follow-up), the conversation history is included in the prompt so the `FINAL_COMPILER` can respond naturally in context rather than receiving a broken empty prompt.

### Clarification (`agent/interaction/clarification.py`)
Resolves ambiguity before planning begins.

**Key functions:**
- `run_clarification(generative_model, chat_history_manager)` — Reads user input and runs the clarification loop; returns `(user_prompt, plan)`.
- `_clarification_loop(...)` — Loops until `plan.needs_clarification=False`, printing each follow-up question and recording answers in history.

### ToolManager (`agent/tools/tool_manager.py`)
Central registry for all agent tools.

**Key methods:**
- `add_tool(tool)` — Registers a `Tool` by name; validates handler callability
- `use_tool(tool_name, **kwargs)` — Dispatches the named tool and returns a `ToolResult`
- `get_all_tools_schema()` — Returns LLM-native schema objects for all registered tools (currently Gemini `genai_types.Tool`)
- `remove_tool(tool_name)` — Deregisters a tool

### CreateTools (`agent/tools/tool_generator.py`)
Factory class with static methods for creating each tool instance.

- `create_rag_tool(rag_handler)` — `rag_query` tool backed by `RAGHandler.get_query_response`
- `create_web_search_tool(web_search_handler)` — `web_search` tool backed by `WebSearchHandler.retrieve_relevant_content_from_web`
- `create_add_data_to_corpus_tool(rag_handler)` — `add_data_to_corpus` tool
- `create_clear_corpus_tool(rag_handler)` — `clear_corpus` tool
- `create_datetime_tool()` — `get_current_datetime` tool
- `create_days_since_epoch_tool()` — `get_days_since_epoch` tool
- `create_internal_knowledge_tool(generative_model)` — `internal_knowledge` tool; accepts a `query` parameter at runtime and calls the generative model with `InternalKnowledge.DIRECT_RESPONSE` to return a real answer rather than a static signal string

### RAGHandler (`agent/tools/tool_handlers/rag_handler.py`)
Owns the full RAG pipeline end-to-end and exposes it as agent-callable tools.

**Key public methods:**
- `get_query_response(query, corpus_name, **kwargs)` — Full pipeline: query rewriting → optional history check → hybrid retrieval → context summarization → response generation → history persistence. Returns a `ToolResult`.
- `retrieve_top_related_items(query, corpus_name, top_k)` — Retrieval-only path; rewrites the query then fetches raw passages without generation. Returns a `ToolResult`.
- `add_data_to_corpus(corpus_name, documents, metadatas)` — Ingests new documents into a named corpus.
- `clear_corpus(corpus_name)` — Wipes all data from a named corpus.

**Key internal methods:**
- `_improve_query(query, **kwargs)` — LLM-powered query rewriting
- `_summarize_context(query, context, **kwargs)` — Query-aware passage summarization
- `_retrieve_relevant_context(query, corpus, **kwargs)` — Wraps the retriever with positive-score filtering
- `_summarize_and_build_contexts_string(query, retrieved_context, ...)` — Summarizes each passage and formats a numbered context string
- `_generate_response_from_context(query, contexts_string, **kwargs)` — Builds the prompt and calls the LLM
- `_add_to_history(query, metadata)` — Indexes interactions for future reuse
- `_check_history(query, similarity_threshold)` — Searches history corpus; returns a hit dict or `None`

### WebSearchHandler (`agent/tools/tool_handlers/web_search_handler.py`)
Integrates live web search with the RAG pipeline.

**Key method:**
- `retrieve_relevant_content_from_web(query, corpus_name, max_web_pages, top_k)` — Optimizes the query for search engines, fetches and extracts content from up to `max_web_pages` URLs via `WebSearch`, indexes them into the named corpus, and returns the top relevant raw passages via `RAGHandler.retrieve_top_related_items`.

> ⚠️ **Disclaimer**
> This tool is provided for educational and research purposes only. It does not enforce `robots.txt` restrictions. Users are responsible for ensuring that their usage complies with the terms of service and `robots.txt` rules of any website they access. The author assumes no responsibility for misuse.

### RAGCorpusManager (`rag/rag_corpus_manager.py`)
Handles document ingestion, chunking, embedding, FAISS indexing, and BM25 indexing.

**Key Features:**
- Sliding-window chunking (configurable `max_data_chunk_len` and `data_chunk_stride`)
- Automatic deduplication via normalized text comparison
- FAISS `IndexFlatIP` (inner product = cosine after L2 normalization) for semantic search
- Integrated `BM25KeywordRetriever` built and updated alongside the FAISS index
- `clear_corpus()` — Resets all stored documents, chunks, embeddings, and indices

### RAGContentRetriever (`rag/rag_content_retriever.py`)
Performs three-stage retrieval: hybrid scoring → cross-encoder re-ranking → chunk merging.

**Retrieval Pipeline:**
1. FAISS bi-encoder search (fast, approximate, semantic)
2. BM25 search (fast, exact keyword matching)
3. Min-max normalization + weighted linear fusion (BM25 step skipped when `lexical_similarity_weight=0.0`)
4. Cross-encoder re-ranking (precise, pairwise scoring)
5. Overlapping chunk merging for coherent passage reconstruction

### BM25KeywordRetriever (`rag/bm25_keyword_retriever.py`)
Lightweight BM25 wrapper with NLTK tokenization and Porter stemming.

**Key Methods:**
- `build_index(documents)` — Builds `BM25Okapi` index from a list of text strings
- `search(query, top_k)` — Returns `(indices, scores)` for top-k relevant documents

### ModelLoader (`utils/model_loader.py`)
Stateless utility for loading all model types.

**Supported Models:**
- `load_sentence_embedding_model()` — `SentenceTransformer` bi-encoder
- `load_hf_cross_encoder()` — `CrossEncoder` re-ranker
- `load_hf_tokenizer()` — `AutoTokenizer`
- `load_gemini_generative_model()` — Returns a `GenerativeModel` dataclass wrapping a Gemini client. The `generate` callable prepends the current date/time to every prompt and accepts `response_schema` for structured Pydantic outputs. Supports an optional `LLMLogger` to write every call to a shared JSON file.
- `load_hf_generative_model()` — Returns a `GenerativeModel` backed by a local Qwen (or other HF causal-LM) model

### ChatHistoryManager (`utils/chat_history_manager.py`)
Manages conversation history with sliding-window access and JSON persistence.

**Key Methods:**
- `append(role, content)` — Adds an entry and writes it to disk
- `get_window_history()` — Returns the most recent `window_size` entries
- `get_history_str()` — Returns the sliding window as a formatted `role: content` string for prompt injection

### LLMLogger (`utils/llm_logger.py`)
Shared per-run logger for all generative model calls.

- One timestamped JSON file is created at instantiation; all model instances sharing the same `LLMLogger` write to the same file
- Each entry contains `timestamp`, `role` (`user`/`agent`), `caller` (the system instruction's `__name__`), and `content`

### DataSetUtil (`utils/dataset_utils.py`)
Utility for loading benchmark datasets.

**Key Method:**
- `load_qa_dataset(dataset_name, samples_count, random_sampling, dataset_split)` — Loads a QA dataset from HuggingFace Hub and returns a list of `{"id", "question", "context", "answers"}` dicts. Defaults to SQuAD `"train"` split.

## 📝 Data Classes (`base/data_classes.py`)

All shared types are declared in one place:

| Class | Type | Purpose |
|---|---|---|
| `ToolResult` | dataclass | Unified return type for every tool handler; `.text` for LLM, `.data` for programmatic use |
| `Tool` | frozen dataclass | Tool descriptor: name, description, parameters, handler |
| `ParameterDetails` | frozen dataclass | Schema metadata for a single tool parameter |
| `GenerativeModel` | dataclass | Provider-agnostic LLM wrapper with a `generate` callable |
| `LLMResponse` | dataclass | Standardized LLM output: `.text`, `.parsed`, `.function_calls` |
| `CorpusInfo` | dataclass | Named corpus descriptor exposed to the RAG handler |
| `ToolCall` | dataclass | Record of one tool invocation in the executor loop |
| `StepResult` | dataclass | Resolved outcome for a single plan step |
| `PlanningAgentResponse` | Pydantic | Structured planner output: goal, steps, clarification state |
| `PlanStep` | Pydantic | A single step inside a plan |
| `InternalSynthesizerResponse` | Pydantic | Executor internal evaluator output: summary, missing_info, early_stop |
| `ExecutorStepResponse` | Pydantic | Executor final synthesizer output: thought, answer |
| `FinalSynthesisResponse` | Pydantic | Final compiler output: final_answer, follow_up_question |

## 🔮 Future Work

### Advanced Planning
- **Parallel Step Execution**: The current orchestrator executes steps sequentially. Steps with no data dependency could run in parallel to reduce total latency.
- **Dynamic Replanning**: If an executor step fails or returns insufficient information, the planner could be re-invoked with the partial results to revise the remaining steps.

### Advanced Query Handling
- **Query Decomposition**: Extend query rewriting to full decomposition — break complex or multi-part questions into independent sub-queries, execute each, and synthesize results
- **Adaptive Routing**: Automatically route queries based on complexity — direct retrieval for simple factual lookups, decomposition for multi-hop questions, web search for time-sensitive information

## 🤝 Contributing

Contributions are welcome!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Semantic embeddings by [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- Vector indexing powered by [FAISS](https://github.com/facebookresearch/faiss) (Meta AI)
- Cross-encoder re-ranking via [Sentence-Transformers](https://www.sbert.net/) (`ms-marco-MiniLM-L-6-v2`)
- Lexical retrieval by [rank_bm25](https://github.com/dorianbrown/rank_bm25) with [NLTK](https://www.nltk.org/) tokenization
- Web search powered by [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) with page extraction via [Trafilatura](https://trafilatura.readthedocs.io/)
- LLM inference via [Google Gemini](https://deepmind.google/technologies/gemini/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- All architecture, design decisions, and code are my own. [GitHub Copilot](https://github.com/features/copilot) helped with writing this README, docstrings, and the occasional refactoring.

## 📧 Contact

**Sahand Mosharafian**
- GitHub: [@sahandmsh](https://github.com/sahandmsh)
- Project Link: [https://github.com/sahandmsh/plan-and-execute-agent](https://github.com/sahandmsh/plan-and-execute-agent)

## 🔖 Citation

If you use this project in your research, please cite:

```bibtex
@software{mosharafian2026planandexecuteagent,
  author = {Mosharafian, Sahand},
  title  = {Plan-and-Execute Agent: A Modular AI Agent with Structured Planning, ReAct Execution, Hybrid RAG, and Live Web Search},
  year   = {2026},
  url    = {https://github.com/sahandmsh/plan-and-execute-agent}
}
```

---

⭐ **Star this repository if you find it helpful!**

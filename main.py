from agent.executor_agent import ExecutorAgent
from agent.interaction.clarification import run_clarification
from agent.orchestrator import run_orchestrator
from agent.tools.tool_generator import CreateTools
from agent.tools.tool_handlers.rag_handler import RAGHandler
from agent.tools.tool_handlers.web_search_handler import WebSearchHandler
from agent.tools.tool_manager import ToolManager
from base.constants import Constants
from base.data_classes import CorpusInfo
from config import HF_TOKEN, GOOGLE_API_KEY
from rag.rag_content_retriever import RAGContentRetriever
from rag.rag_corpus_manager import RAGCorpusManager
from utils.chat_history_manager import ChatHistoryManager
from utils.dataset_utils import DataSetUtil
from utils.llm_logger import LLMLogger
from utils.model_loader import ModelLoader


if __name__ == "__main__":
    # ============================================================================
    # STEP 1: Load Models
    # ============================================================================

    model_loader = ModelLoader()

    # Shared LLM logger — writes all generative model calls to a timestamped JSON file
    llm_logger = LLMLogger(log_dir="./logs")

    # Cross-encoder used for re-ranking candidate passages during retrieval
    cross_encoder = model_loader.load_hf_cross_encoder(hugging_face_token=HF_TOKEN)

    # Sentence transformer used to produce dense embeddings for semantic search
    sentence_transformer = model_loader.load_sentence_embedding_model(hugging_face_token=HF_TOKEN)

    # Main generative model shared across all agents (planning, execution, synthesis)
    generative_model = model_loader.load_gemini_generative_model(
        google_api_key=GOOGLE_API_KEY,
        config=None,
        model_name=Constants.ModelNames.Gemini.GEMINI_2_5_FLASH,
        llm_logger=llm_logger,
    )

    # ============================================================================
    # STEP 2: Build RAG Corpora and Index the Knowledge Base
    # ============================================================================

    # Corpus for the static knowledge base (local documents / SQuAD)
    rag_corpus_manager_for_knowledge_base = RAGCorpusManager(
        sentence_transformer=sentence_transformer
    )

    # Corpus for the RAG agent's internal session history (used for history-aware retrieval)
    rag_corpus_manager_for_rag_agent_history = RAGCorpusManager(
        sentence_transformer=sentence_transformer
    )

    # Temporary corpus populated at runtime with live web search results
    rag_corpus_manager_for_web_search = RAGCorpusManager(sentence_transformer=sentence_transformer)

    # Load and index the knowledge base (SQuAD QA dataset, 10000 samples)
    dataset_util = DataSetUtil()
    knowledge_base = dataset_util.load_qa_dataset(samples_count=10000)

    rag_corpus_manager_for_knowledge_base.add_update_data_and_index(
        texts=[item["context"] for item in knowledge_base],
        metadatas=[{k: v for k, v in item.items() if k != "context"} for item in knowledge_base],
    )

    # Content retriever wraps the corpus managers and uses the cross-encoder to re-rank results
    rag_content_retriever = RAGContentRetriever(cross_encoder)

    # ============================================================================
    # STEP 3: Initialize the RAG Handler
    # ============================================================================

    # Corpus descriptors expose each corpus to the RAG handler with a name and description
    knowledge_base_corpus = CorpusInfo(
        name="knowledge_base",
        corpus=rag_corpus_manager_for_knowledge_base,
        description="General knowledge base indexed from the SQuAD dataset, covering a wide range of topics. Useful for answering questions when the user asks about retrieving answers from documents.",
    )
    web_search_corpus = CorpusInfo(
        name="web_search",
        corpus=rag_corpus_manager_for_web_search,
        description="Temporary corpus populated with live web search results for real-time or up-to-date queries.",
    )

    # RAG handler owns the full pipeline: retrieval, rewriting, summarization,
    # response generation, and session-history caching.
    rag_handler = RAGHandler(
        history_corpus=rag_corpus_manager_for_rag_agent_history,
        rag_retriever=rag_content_retriever,
        rag_corpus_dict={
            knowledge_base_corpus.name: knowledge_base_corpus,
            web_search_corpus.name: web_search_corpus,
        },
        generative_model=generative_model,
    )

    # Web search handler fetches live results and ingests them into the web-search corpus
    web_search_handler = WebSearchHandler(rag_handler=rag_handler)

    # ============================================================================
    # STEP 4: Create Tools and Register Them with the Tool Manager
    # ============================================================================

    tool_generator = CreateTools()

    # Date/time utilities
    datetime_tool = tool_generator.create_datetime_tool()
    days_since_epoch_tool = tool_generator.create_days_since_epoch_tool()

    # RAG-backed retrieval and corpus management tools
    rag_tool = tool_generator.create_rag_tool(rag_handler)
    add_data_to_corpus_tool = tool_generator.create_add_data_to_corpus_tool(rag_handler)
    clear_corpus_tool = tool_generator.create_clear_corpus_tool(rag_handler)

    # Live web search tool (fetches results and adds them to the web-search corpus)
    web_search_tool = tool_generator.create_web_search_tool(web_search_handler)

    # Fallback tool that uses the model's parametric (internal) knowledge directly
    internal_knowledge_tool = tool_generator.create_internal_knowledge_tool(generative_model)

    tool_manager = ToolManager(Constants.Tools.SupportedLLMs.GEMINI)
    tool_manager.add_tool(datetime_tool)
    tool_manager.add_tool(days_since_epoch_tool)
    tool_manager.add_tool(rag_tool)
    tool_manager.add_tool(add_data_to_corpus_tool)
    tool_manager.add_tool(clear_corpus_tool)
    tool_manager.add_tool(web_search_tool)
    tool_manager.add_tool(internal_knowledge_tool)

    # ============================================================================
    # STEP 5: Initialize the Executor Agent and Chat History Manager
    # ============================================================================

    # Executor runs each plan step via a ReAct loop, dispatching tool calls as needed
    executor = ExecutorAgent(
        generative_model=generative_model,
        tool_manager=tool_manager,
    )

    # Chat history persists the full conversation to JSON and provides a
    # sliding-window view to keep the LLM context bounded
    chat_history_manager = ChatHistoryManager(log_dir="./logs", window_size=100)

    # ============================================================================
    # STEP 6: Interactive Loop
    # ============================================================================

    print("\n===== Modular Planning Agent Interactive Mode =====")
    print("Type your prompt and press Enter. Press Ctrl+C to exit.\n")
    while True:
        try:
            # Collect the user's request, optionally clarifying ambiguities,
            # and produce a structured plan
            user_prompt, plan = run_clarification(
                generative_model=generative_model,
                chat_history_manager=chat_history_manager,
            )
            if not user_prompt:
                continue

            # Execute the plan step-by-step and synthesize the final response
            run_orchestrator(
                user_prompt=user_prompt,
                plan=plan,
                generative_model=generative_model,
                executor=executor,
                chat_history_manager=chat_history_manager,
            )
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

from agent.tools.tool_handlers.rag_handler import RAGHandler
from base.data_classes import LLMResponse, ToolResult
from base.constants import Constants
from utils.web_search import WebSearch


class WebSearchHandler:
    """Handler for the web search tool, which integrates external web search results into the RAG retrieval process.

    Attributes:
        web_search (WebSearch): Component used to execute external web queries.
        rag_corpus (RAGCorpusManager): Manages temporary storage/indexing of web results.
        rag_content_retriever (RAGContentRetriever): Performs similarity search and ranking.
    """

    def __init__(self, rag_handler: RAGHandler):
        """Initialize the web search tool.

        Args:
            rag_handler (RAGHandler): Retrieval component for ranking passages.
        """
        self.web_search = WebSearch()
        self.rag_handler = rag_handler
        self.generative_model = rag_handler.generative_model

    def retrieve_relevant_content_from_web(
        self, query: str, corpus_name: str, max_web_pages: int = 5, top_k: int = None
    ) -> ToolResult:
        """Execute web search + retrieval-only RAG for a query to find relevant web passages.

        Process:
            1. Optimise the query for a search engine.
            2. Fetch up to `max_web_pages` web results and index them into the corpus.
            3. Run similarity search / re-ranking and return the top passages as-is.
               No query rewriting, context summarization, or answer generation is performed —
               that is left to the downstream synthesizer.
        Args:
            query (str): The search query string.
            corpus_name (str): Name of the RAG corpus to use for indexing/retrieval.
            max_web_pages (int, optional): Maximum number of web pages to retrieve and index. Default is 3.
            top_k (int, optional): Number of top passages to return. Defaults to the RAG handler's
                                   ``top_k`` setting if not provided.
        Returns:
            ToolResult: .text is a numbered list of the most relevant web passages for the LLM;
                        .data is the raw list of passage dicts for programmatic use.
        Raises:
            ValueError: If the specified corpus name is not found in the RAG handler's corpus dictionary.
        """
        corpus_config = self.rag_handler.rag_corpus_dict.get(corpus_name, None)
        if not corpus_config:
            raise ValueError(
                f"Corpus '{corpus_name}' not found in RAG handler's corpus dictionary."
            )
        web_search_query: LLMResponse = self.generative_model.generate(
            query,
            system_instructions=Constants.Instructions.WebSearch.QUERY_OPTIMIZER,
        ).text
        try:
            web_documents, web_documents_metadata = self.web_search.search(
                web_search_query, max_results=max_web_pages
            )
        except Exception as e:
            return ToolResult(
                text=f"Web search failed: {str(e)}",
                data={"error": str(e)},
            )
        corpus_config.corpus.add_update_data_and_index(web_documents, web_documents_metadata)
        top_k = top_k if top_k is not None else self.rag_handler.top_k
        tool_result: ToolResult = self.rag_handler.retrieve_top_related_items(
            query=query, corpus_name=corpus_name, top_k=top_k
        )
        # Prepend a note so downstream planner/synthesizer knows the corpus is now populated
        tool_result.text = (
            f"[Web search complete. Results indexed into corpus '{corpus_name}'. "
            f"Future follow-up questions on this topic can use RAG"
            f"against corpus '{corpus_name}' without re-fetching.]\n\n" + tool_result.text
        )
        return tool_result

from base.constants import Constants
from base.data_classes import (
    CorpusInfo,
    GenerativeModel,
    LLMResponse,
    ToolResult,
)
from rag.rag_content_retriever import RAGContentRetriever
from rag.rag_corpus_manager import RAGCorpusManager


class RAGHandler:
    """Handles the full RAG pipeline and exposes it as agent-callable tools.

    This class owns all RAG logic end-to-end: query rewriting, hybrid retrieval,
    cross-encoder re-ranking, context summarization, grounded response generation,
    and session-history caching.  It also provides a retrieval-only path for
    downstream steps that need raw passages rather than a generated answer.
    """

    def __init__(
        self,
        history_corpus: RAGCorpusManager,
        rag_retriever: RAGContentRetriever,
        rag_corpus_dict: dict[str, CorpusInfo],
        generative_model: GenerativeModel,
        similarity_score_threshold: float = 6.5,
        semantic_similarity_weight: float = 0.8,
        lexical_similarity_weight: float = 0.2,
        top_k: int = 5,
        use_agent_history: bool = False,
    ):
        """Initialize the RAG handler.

        Args:
            history_corpus (RAGCorpusManager): Corpus used to cache and retrieve
                past query/response pairs for history-aware retrieval.
            rag_retriever (RAGContentRetriever): Retriever that performs hybrid
                (dense + BM25) search and cross-encoder re-ranking.
            rag_corpus_dict (dict[str, CorpusInfo]): Named corpora available for
                retrieval, keyed by corpus name.
            generative_model (GenerativeModel): The generative model.
            similarity_score_threshold (float): Minimum cross-encoder score to
                accept a history hit as valid.  Scores are unbounded but typically
                fall in [-10, 10].  Defaults to 6.5.
            semantic_similarity_weight (float): Weight for dense retrieval (0–1).
            lexical_similarity_weight (float): Weight for sparse/BM25 retrieval (0–1).
            top_k (int): Number of passages to keep after re-ranking.
            use_agent_history (bool): Whether to check session history before
                performing a fresh retrieval.  Defaults to False.
        """
        self.history_corpus = history_corpus
        self.rag_retriever = rag_retriever
        self.rag_corpus_dict = rag_corpus_dict
        self.generative_model = generative_model
        self.similarity_score_threshold = similarity_score_threshold
        self.semantic_similarity_weight = semantic_similarity_weight
        self.lexical_similarity_weight = lexical_similarity_weight
        self.top_k = top_k
        self.use_agent_history = use_agent_history

    # -------------------------------------------------------------------------
    # Internal helpers (RAG pipeline steps)
    # -------------------------------------------------------------------------

    def _available_corpora_description(self) -> str:
        """Return a human-readable listing of registered corpora for use in tool descriptions.

        Returns:
            str: Semicolon-separated entries of the form ``'<name>': <description>``.
                 Example: ``'kb': General knowledge base; 'finance': Financial reports``
        """
        return "; ".join(
            f"'{corpus_config.name}': {corpus_config.description}"
            for corpus_config in self.rag_corpus_dict.values()
        )

    def _improve_query(self, query: str, **kwargs) -> LLMResponse:
        """Rewrite the user query to be clearer and better suited for retrieval.

        Uses ``Constants.Instructions.RAG.QUERY_REWRITER`` as the system
        instruction.

        Args:
            query (str): The original user query.
            **kwargs: Extra keyword arguments forwarded to the generative model
                (e.g. a faster/cheaper ``config``).

        Returns:
            LLMResponse: The rewritten query in ``.text``.
        """
        return self.generative_model.generate(
            f"Query: {query}",
            system_instructions=Constants.Instructions.RAG.QUERY_REWRITER,
            **kwargs,
        )

    def _summarize_context(self, query: str, context: str, **kwargs) -> LLMResponse:
        """Produce a concise, query-focused summary of a single retrieved passage.

        Uses ``Constants.Instructions.RAG.CONTEXT_SUMMARIZER`` as the
        system instruction.

        Args:
            query (str): The user query that guides the summarization focus.
            context (str): A single retrieved passage to summarize.
            **kwargs: Extra keyword arguments forwarded to the generative model.

        Returns:
            LLMResponse: The summary in ``.text``.
        """
        return self.generative_model.generate(
            f"Query: {query}\nContext:\n{context}",
            system_instructions=Constants.Instructions.RAG.CONTEXT_SUMMARIZER,
            **kwargs,
        )

    def _retrieve_relevant_context(
        self,
        query: str,
        corpus: RAGCorpusManager,
        **retriever_kwargs,
    ) -> tuple[list[str], list[dict]]:
        """Retrieve relevant passages for a query from the given corpus.

        Args:
            query (str): The (optionally rewritten) query.
            corpus (RAGCorpusManager): The corpus to search.
            **retriever_kwargs: Extra kwargs forwarded to the retriever
                (e.g. ``top_k``, ``semantic_similarity_weight``).

        Returns:
            tuple[list[str], list[dict]]: Retrieved passages and their metadata.
        """
        indices, scores = self.rag_retriever.find_top_similar_items(
            rag_corpus=corpus, query=query, **retriever_kwargs
        )
        positive_mask = [score > 0 for score in scores]
        indices = [idx for idx, keep in zip(indices, positive_mask) if keep]
        if not indices:
            return [], []
        retrieved_context = [corpus.chunked_data[idx] for idx in indices]
        retrieved_context_metadata = [corpus.chunked_data_metadata[idx] for idx in indices]
        return retrieved_context, retrieved_context_metadata

    def _summarize_and_build_contexts_string(
        self,
        query: str,
        retrieved_context: list[str],
        summarize_context_kwargs: dict | None = None,
    ) -> tuple[list[str], str]:
        """Summarize each retrieved passage and format them into a numbered context string.

        Args:
            query (str): The user query, used to focus each summary.
            retrieved_context (list[str]): Raw retrieved passages.
            summarize_context_kwargs (dict | None): Extra kwargs forwarded to the
                generative model inside ``_summarize_context``.  Defaults to None.

        Returns:
            tuple[list[str], str]: The list of summarized passages and the
                formatted numbered context string ready to inject into a prompt.
        """
        if not retrieved_context:
            return [], "Context: "
        summarized_context = [
            self._summarize_context(query, passage, **(summarize_context_kwargs or {})).text
            for passage in retrieved_context
        ]
        contexts_string = "\n\n".join(
            f"Context {i + 1}: {passage}" for i, passage in enumerate(summarized_context)
        )
        return summarized_context, contexts_string

    def _generate_response_from_context(
        self,
        query: str,
        contexts_string: str,
        **generative_model_kwargs,
    ) -> LLMResponse:
        """Build the prompt and call the generative model to produce a grounded response.

        Args:
            query (str): The user query.
            contexts_string (str): Formatted numbered context string.
            **generative_model_kwargs: Extra kwargs forwarded to the generative
                model (e.g. ``config``).

        Returns:
            LLMResponse: The model response.
        """
        return self.generative_model.generate(
            f"Query: {query}\n{contexts_string}",
            system_instructions=Constants.Instructions.RAG.RESPONSE_GENERATOR,
            **generative_model_kwargs,
        )

    def _add_to_history(self, query: str, metadata: dict) -> None:
        """Persist a query/response pair in the session history corpus.

        Queries are stored as the searchable text; responses and other details
        live in the metadata.

        Args:
            query (str): The (improved) query to store.
            metadata (dict): Response details to attach as metadata.
        """
        self.history_corpus.add_update_data_and_index(texts=[query], metadatas=[metadata])

    def _check_history(self, query: str, similarity_threshold: float) -> dict | None:
        """Look up the session history corpus for a sufficiently similar past response.

        Args:
            query (str): The (improved) query to check against history.
            similarity_threshold (float): Minimum score to consider a hit valid.

        Returns:
            dict | None: History hit metadata dict if a match is found, else None.
        """
        indices, scores = self.rag_retriever.find_top_similar_items(
            rag_corpus=self.history_corpus,
            query=query,
            top_k=1,
        )
        if scores and scores[0] >= similarity_threshold:
            meta = self.history_corpus.chunked_data_metadata[indices[0]].copy()
            meta.pop(Constants.CorpusManager.MetadataKeys.DATA_CHUNK_ID, None)
            meta.pop(Constants.CorpusManager.MetadataKeys.DOCUMENT_ID, None)
            history_query = meta.pop("query", None)
            return {"relevance_score": scores[0], "history_query": history_query, **meta}
        return None

    # -------------------------------------------------------------------------
    # Tool-facing public methods
    # -------------------------------------------------------------------------

    def get_query_response(self, query: str, corpus_name: str, **kwargs) -> ToolResult:
        """Run the full end-to-end RAG pipeline and return a grounded answer.

        Steps: query rewriting → optional history check → hybrid retrieval →
        context summarization → response generation → history persistence.

        Args:
            query (str): The user's question.
            corpus_name (str): Name of the corpus to query (must be a key in
                ``rag_corpus_dict``).
            **kwargs: Optional runtime overrides for any pipeline parameter:
                ``similarity_score_threshold``, ``semantic_similarity_weight``,
                ``lexical_similarity_weight``, ``top_k``, ``use_agent_history``,
                ``rag_retriever_kwargs_dict``, ``improve_query_kwargs_dict``,
                ``summarize_context_kwargs_dict``, ``generate_response_kwargs_dict``.

        Returns:
            ToolResult: ``.text`` is the LLM-ready answer string;
                        ``.data`` is the full response dict for programmatic use.
        """
        corpus = self.rag_corpus_dict.get(corpus_name)
        if corpus is None:
            raise ValueError(f"Corpus '{corpus_name}' not found in RAG corpus dictionary.")

        similarity_score_threshold = kwargs.get(
            "similarity_score_threshold", self.similarity_score_threshold
        )
        semantic_similarity_weight = kwargs.get(
            "semantic_similarity_weight", self.semantic_similarity_weight
        )
        lexical_similarity_weight = kwargs.get(
            "lexical_similarity_weight", self.lexical_similarity_weight
        )
        top_k = kwargs.get("top_k", self.top_k)
        use_agent_history = kwargs.get("use_agent_history", self.use_agent_history)
        rag_retriever_kwargs = kwargs.get("rag_retriever_kwargs_dict") or {}
        improve_query_kwargs = kwargs.get("improve_query_kwargs_dict") or {}
        summarize_context_kwargs = kwargs.get("summarize_context_kwargs_dict") or {}
        generate_response_kwargs = kwargs.get("generate_response_kwargs_dict") or {}

        improved_query = self._improve_query(query, **improve_query_kwargs).text

        # --- 1. History lookup ---
        if use_agent_history:
            history_hit = self._check_history(improved_query, similarity_score_threshold)
            if history_hit is not None:
                response_details_dict = {
                    **history_hit,
                    "query": query,
                    "improved_query": improved_query,
                    "response_type": "agent_history",
                    "agent_history_metadata": {
                        "relevance_score": history_hit["relevance_score"],
                        "history_query": history_hit["history_query"],
                    },
                }
                parsed = response_details_dict.get("parsed_response") or {}
                llm_text = (
                    parsed.get("response") or response_details_dict.get("response_text") or ""
                )
                return ToolResult(text=llm_text, data=response_details_dict)

        # --- 2. Fresh retrieval ---
        retrieved_context, _ = self._retrieve_relevant_context(
            improved_query,
            corpus.corpus,
            top_k=top_k,
            semantic_similarity_weight=semantic_similarity_weight,
            lexical_similarity_weight=lexical_similarity_weight,
            **rag_retriever_kwargs,
        )

        # --- 3. Summarize context and generate response ---
        summarized_context, contexts_string = self._summarize_and_build_contexts_string(
            query=improved_query,
            retrieved_context=retrieved_context,
            summarize_context_kwargs=summarize_context_kwargs,
        )
        response = self._generate_response_from_context(
            query, contexts_string, **generate_response_kwargs
        )

        # --- 4. Persist and return ---
        response_details_dict = {
            "query": query,
            "improved_query": improved_query,
            "response_type": "generative_model",
            "agent_history_metadata": None,
            "context": summarized_context,
            "response_text": response.text,
            "parsed_response": (
                response.parsed.model_dump()
                if hasattr(response.parsed, "model_dump")
                else response.parsed
            ),
        }
        self._add_to_history(improved_query, response_details_dict)

        parsed = response_details_dict.get("parsed_response") or {}
        llm_text = parsed.get("response") or response_details_dict.get("response_text") or ""
        return ToolResult(text=llm_text, data=response_details_dict)

    def add_data_to_corpus(
        self, corpus_name: str, documents: list[str], metadatas: list[dict] = None
    ) -> ToolResult:
        """Add new documents to the specified corpus and re-index it.

        Args:
            corpus_name (str): The name of the corpus to update.
            documents (list[str]): Documents to add, one string per document.
            metadatas (list[dict], optional): Metadata dicts, one per document.

        Returns:
            ToolResult: Confirmation message.
        """
        corpus = self.rag_corpus_dict.get(corpus_name)
        if corpus is None:
            raise ValueError(f"Corpus '{corpus_name}' not found in RAG corpus dictionary.")
        corpus.corpus.add_update_data_and_index(documents, metadatas)
        return ToolResult(text=f"Corpus '{corpus_name}' has been successfully updated.")

    def clear_corpus(self, corpus_name: str) -> ToolResult:
        """Clear all data from the specified corpus.

        Args:
            corpus_name (str): The name of the corpus to clear.

        Returns:
            ToolResult: Confirmation message that the corpus was cleared.

        Raises:
            ValueError: If the specified corpus name is not found in the RAG corpus dictionary.
        """
        corpus = self.rag_corpus_dict.get(corpus_name)
        if corpus is None:
            raise ValueError(f"Corpus '{corpus_name}' not found in RAG corpus dictionary.")
        corpus.corpus.clear_corpus()
        return ToolResult(text=f"Corpus '{corpus_name}' has been successfully cleared.")

    def retrieve_top_related_items(
        self, query: str, corpus_name: str, top_k: int = 5
    ) -> ToolResult:
        """Retrieve the top-k most relevant passages from the corpus for a given query.

        This is a **retrieval-only** operation — it does NOT rewrite the query,
        summarize the context, or generate a final answer.  Use this when raw
        passages are needed directly (e.g. for exploration or downstream processing).
        For a full end-to-end pipeline (query rewriting → context summarization →
        grounded response), use ``get_query_response`` instead.

        Args:
            query (str): The question or topic to find related passages for.
            corpus_name (str): The name of the corpus to search within.
            top_k (int): Number of top relevant passages to retrieve.  Default is 5.

        Returns:
            ToolResult: ``.text`` is a numbered list of the retrieved passages;
                        ``.data`` is the raw list of passage strings.
        """
        corpus = self.rag_corpus_dict.get(corpus_name)
        if corpus is None:
            raise ValueError(f"Corpus '{corpus_name}' not found in RAG corpus dictionary.")
        optimized_query = self._improve_query(query).text
        top_items, _ = self._retrieve_relevant_context(
            query=optimized_query, corpus=corpus.corpus, top_k=top_k
        )
        if not top_items:
            return ToolResult(text="No related passages found for the given query.", data=[])
        lines = [f"[{i + 1}] {item}" for i, item in enumerate(top_items)]
        llm_text = (
            f"Top {len(top_items)} related passage(s) retrieved from corpus '{corpus_name}':\n"
            + "\n".join(lines)
        )
        return ToolResult(text=llm_text, data=top_items)

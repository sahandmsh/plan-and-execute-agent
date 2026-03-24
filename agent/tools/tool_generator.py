from base.constants import Constants
from base.data_classes import ParameterDetails, Tool, ToolResult
from agent.tools.tool_handlers.datetime_handler import (
    get_current_datetime,
    get_days_since_epoch,
)
from agent.tools.tool_handlers.rag_handler import RAGHandler
from agent.tools.tool_handlers.web_search_handler import WebSearchHandler


class CreateTools:
    """A centralized class for creating Tool instances."""

    @staticmethod
    def create_internal_knowledge_tool() -> Tool:
        """Create the internal knowledge tool.

        Returns:
            Tool: The configured internal knowledge Tool object.
        """
        return Tool(
            name=Constants.Tools.InternalKnowledgeTool.NAME,
            description=Constants.Tools.InternalKnowledgeTool.DESCRIPTION,
            parameters={},
            handler=lambda: ToolResult(
                text="Use internal knowledge to answer."
            ),  # Simple handler for demonstration
        )

    @staticmethod
    def create_datetime_tool() -> Tool:
        """Create the datetime tool.

        Returns:
            Tool: The configured datetime Tool object.
        """
        return Tool(
            name=Constants.Tools.DatetimeTool.NAME,
            description=Constants.Tools.DatetimeTool.DESCRIPTION,
            parameters={},
            handler=get_current_datetime,
        )

    @staticmethod
    def create_days_since_epoch_tool() -> Tool:
        """Create the days since epoch tool.

        Returns:
            Tool: The configured days since epoch Tool object.
        """
        return Tool(
            name=Constants.Tools.DaysSinceEpochTool.NAME,
            description=Constants.Tools.DaysSinceEpochTool.DESCRIPTION,
            parameters={
                Constants.Tools.DaysSinceEpochTool.Parameters.TargetDate.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=Constants.Tools.DaysSinceEpochTool.Parameters.TargetDate.DESCRIPTION,
                    required=True,
                )
            },
            handler=get_days_since_epoch,
        )

    @staticmethod
    def create_rag_tool(rag_handler: RAGHandler) -> Tool:
        """Create the RAG tool with required 'corpus_name' parameter."""
        parameters = {
            Constants.Tools.RagTool.Parameters.Query.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.STRING,
                description=Constants.Tools.RagTool.Parameters.Query.DESCRIPTION,
                required=True,
            ),
            Constants.Tools.RagTool.Parameters.CorpusName.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.STRING,
                description=(
                    f"{Constants.Tools.RagTool.Parameters.CorpusName.DESCRIPTION}. "
                    f"Available corpora: {rag_handler._available_corpora_description()}"
                ),
                required=True,
            ),
            Constants.Tools.RagTool.Parameters.SimilarityScoreThreshold.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.FLOAT,
                description=Constants.Tools.RagTool.Parameters.SimilarityScoreThreshold.DESCRIPTION,
                required=False,
            ),
            Constants.Tools.RagTool.Parameters.SemanticSimilarityWeight.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.FLOAT,
                description=Constants.Tools.RagTool.Parameters.SemanticSimilarityWeight.DESCRIPTION,
                required=False,
            ),
            Constants.Tools.RagTool.Parameters.LexicalSimilarityWeight.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.FLOAT,
                description=Constants.Tools.RagTool.Parameters.LexicalSimilarityWeight.DESCRIPTION,
                required=False,
            ),
            Constants.Tools.RagTool.Parameters.CrossEncoderTopK.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.INTEGER,
                description=Constants.Tools.RagTool.Parameters.CrossEncoderTopK.DESCRIPTION,
                required=False,
            ),
            Constants.Tools.RagTool.Parameters.UseAgentHistory.NAME: ParameterDetails(
                type=Constants.Tools.Parameters.Types.BOOLEAN,
                description=Constants.Tools.RagTool.Parameters.UseAgentHistory.DESCRIPTION,
                required=False,
            ),
        }

        return Tool(
            name=Constants.Tools.RagTool.NAME,
            description=Constants.Tools.RagTool.DESCRIPTION,
            parameters=parameters,
            handler=rag_handler.get_query_response,
        )

    @staticmethod
    def create_add_data_to_corpus_tool(rag_handler: RAGHandler) -> Tool:
        """Create the add-data-to-corpus tool.

        Returns:
            Tool: The configured add-data-to-corpus Tool object.
        """
        return Tool(
            name=Constants.Tools.AddDataToCorpusTool.NAME,
            description=Constants.Tools.AddDataToCorpusTool.DESCRIPTION,
            parameters={
                Constants.Tools.AddDataToCorpusTool.Parameters.CorpusName.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=(
                        f"{Constants.Tools.AddDataToCorpusTool.Parameters.CorpusName.DESCRIPTION} "
                        f"Available corpora: {rag_handler._available_corpora_description()}."
                    ),
                    required=True,
                ),
                Constants.Tools.AddDataToCorpusTool.Parameters.Documents.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.ARRAY,
                    description=Constants.Tools.AddDataToCorpusTool.Parameters.Documents.DESCRIPTION,
                    required=True,
                    items_type=Constants.Tools.Parameters.Types.STRING,
                ),
                Constants.Tools.AddDataToCorpusTool.Parameters.Metadatas.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.ARRAY,
                    description=Constants.Tools.AddDataToCorpusTool.Parameters.Metadatas.DESCRIPTION,
                    required=False,
                    items_type=Constants.Tools.Parameters.Types.OBJECT,
                ),
            },
            handler=rag_handler.add_data_to_corpus,
        )

    @staticmethod
    def create_clear_corpus_tool(rag_handler: RAGHandler) -> Tool:
        """Create the clear-corpus tool.

        Returns:
            Tool: The configured clear-corpus Tool object.
        """
        return Tool(
            name=Constants.Tools.ClearCorpusTool.NAME,
            description=Constants.Tools.ClearCorpusTool.DESCRIPTION,
            parameters={
                Constants.Tools.ClearCorpusTool.Parameters.CorpusName.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=(
                        f"{Constants.Tools.ClearCorpusTool.Parameters.CorpusName.DESCRIPTION} "
                        f"Available corpora: {rag_handler._available_corpora_description()}."
                    ),
                    required=True,
                ),
            },
            handler=rag_handler.clear_corpus,
        )

    @staticmethod
    def create_related_topics_retriever_tool(rag_handler: RAGHandler) -> Tool:
        """Create the related-topics retriever tool.

        This tool exposes **retrieval only** — no context
        summarization, or response generation.  It is distinct from
        ``create_rag_tool`` (``rag_query``), which runs the full end-to-end
        RAG pipeline.  Use this when raw passages are needed directly.

        Returns:
            Tool: The configured related-topics-retriever Tool object.
        """
        return Tool(
            name=Constants.Tools.RelatedTopicsRetrieverTool.NAME,
            description=(
                f"{Constants.Tools.RelatedTopicsRetrieverTool.DESCRIPTION} "
                f"Available corpora: {rag_handler._available_corpora_description()}"
            ),
            parameters={
                Constants.Tools.RelatedTopicsRetrieverTool.Parameters.Query.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=Constants.Tools.RelatedTopicsRetrieverTool.Parameters.Query.DESCRIPTION,
                    required=True,
                ),
                Constants.Tools.RelatedTopicsRetrieverTool.Parameters.CorpusName.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=(
                        f"{Constants.Tools.RelatedTopicsRetrieverTool.Parameters.CorpusName.DESCRIPTION} "
                        f"Available corpora: {rag_handler._available_corpora_description()}"
                    ),
                    required=True,
                ),
                Constants.Tools.RelatedTopicsRetrieverTool.Parameters.TopK.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.INTEGER,
                    description=Constants.Tools.RelatedTopicsRetrieverTool.Parameters.TopK.DESCRIPTION,
                    required=False,
                ),
            },
            handler=rag_handler.retrieve_top_related_items,
        )

    @staticmethod
    def create_web_search_tool(web_search_handler: WebSearchHandler) -> Tool:
        """Create the web search tool.

        Returns:
            Tool: The configured web search Tool object.
        """
        return Tool(
            name=Constants.Tools.WebSearchTool.NAME,
            description=Constants.Tools.WebSearchTool.DESCRIPTION,
            parameters={
                Constants.Tools.WebSearchTool.Parameters.Query.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=Constants.Tools.WebSearchTool.Parameters.Query.DESCRIPTION,
                    required=True,
                ),
                Constants.Tools.WebSearchTool.Parameters.CorpusName.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.STRING,
                    description=(
                        f"{Constants.Tools.WebSearchTool.Parameters.CorpusName.DESCRIPTION} "
                        f"Available corpora: {web_search_handler.rag_handler._available_corpora_description()}."
                    ),
                    required=True,
                ),
                Constants.Tools.WebSearchTool.Parameters.MaxWebPages.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.INTEGER,
                    description=Constants.Tools.WebSearchTool.Parameters.MaxWebPages.DESCRIPTION,
                    required=False,
                ),
                Constants.Tools.WebSearchTool.Parameters.TopK.NAME: ParameterDetails(
                    type=Constants.Tools.Parameters.Types.INTEGER,
                    description=Constants.Tools.WebSearchTool.Parameters.TopK.DESCRIPTION,
                    required=False,
                ),
            },
            handler=web_search_handler.retrieve_relevant_content_from_web,
        )

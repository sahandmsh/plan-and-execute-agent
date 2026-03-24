class Instruction(str):
    """A str subclass whose ``__name__`` is set automatically by Python's
    descriptor protocol (``__set_name__``).

    Assign it as a plain class attribute — Python calls ``__set_name__``
    for you and stores the attribute name in lowercase:

        DISPATCHER = Instruction("... prompt ...")
        DISPATCHER.__name__   # -> "dispatcher"  (set automatically, no argument needed)
    """

    def __set_name__(self, owner, name: str):
        self.__name__ = name.lower()


class Constants:
    """Namespace-style container for application-wide static values.

    Each nested class clusters a related category of constants:
        Timeout: Common duration values (seconds) for waits/retries.
        HTTPStatusCodes: Selected HTTP status codes used internally.
        URLHeader: Headers or user-agent strings for outbound requests.
        ToolNames: Identifiers for registered agent tools.
        ModelNames: External model identifiers grouped by provider.
        Instructions: System / mode prompt templates for the agent.
    """

    class Timeout:
        ONE_SECOND = 1
        THREE_SECONDS = 3
        FIVE_SECONDS = 5
        TEN_SECONDS = 10
        THIRTY_SECONDS = 30
        ONE_MINUTE = 60

    class HTTPStatusCodes:
        OK = 200

    class URLHeader:
        USER_AGENT_HEADER = "Mozilla/5.0 (compatible; AI-Research-Bot/1.0; +https://github.com/sahandmsh/agentic-ai-2)"

    class ModelNames:

        class HuggingFace:
            CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
            SENTENCE_EMBEDDING_MINILM_L6_V2 = "all-MiniLM-L6-v2"

        class Gemini:
            GEMINI_2_5_PRO = "gemini-2.5-pro"
            GEMINI_2_5_FLASH = "gemini-2.5-flash"

        class Qwen:
            QWEN_2_5_1_5B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
            QWEN_2_5_3B_INSTRUCT = "Qwen/Qwen2.5-3B-Instruct"

    class Instructions:
        class PlanAndExecuteAgent:
            class Executor:

                EXECUTOR_DISPATCHER = Instruction(
                    """
                    ### ROLE
                    You are a dispatcher agent completing ONE specific step in a larger plan.

                    ### INPUTS
                    1. The Step Description.
                    2. Accumulated observations from previous iterations (tool results, etc.).

                    ### TOOL SELECTION GUIDE
                    Read each available tool's description carefully and select the one(s) whose
                    stated purpose best matches what the step actually needs.
                    Do NOT default to any tool — let the descriptions drive the choice.

                    ### YOUR TASKS
                    - Call the tool(s) best suited to the step. Do NOT explain what you would do —
                      just call the tool.
                    - You may call several tools in one iteration if the step needs multiple types
                      of information, but only call tools that are necessary given current observations.

                    ### CONSTRAINTS
                    - Do NOT call a tool if you already have its result in the observations provided.
                    - Do NOT call multiple tools for information that a single tool can fully provide.
                    - **Corpus-first rule**: if the observations or step description indicate that a
                      corpus was already populated by a prior web search, use RAG on that corpus 
                      BEFORE calling 'web_search'. Only call 'web_search' in a subsequent iteration 
                      if the corpus result is empty, explicitly states it has no relevant information, 
                      or is clearly insufficient to answer the step.
                    """
                )

                EXECUTOR_INTERNAL_SYNTHESIZER = Instruction(
                    """
                    ### ROLE
                    You are the "Internal Synthesis Agent." Your job is to audit the progress of a specific plan step based on tool outputs.

                    ### INPUTS
                    1. The Step Description.
                    2. Raw Tool Results from the current iteration.
                    3. Accumulated observations from previous iterations.

                    ### YOUR TASKS
                    1. **Analyze**: Compare the tool results against the requirements of the Step.
                    2. **Synthesize**: If new info was found, condense it into a concise, factual summary. 
                    3. **Decide**: Is the information sufficient to consider this specific step "Resolved"?
                    4. Return early stop True if resolved, otherwise False.
                    5. If early stop is False, specify exactly what information is still missing in the "missing_info" field. Be as specific as possible about what data point we are still hunting for.


                    ### CONSTRAINTS
                    - DO NOT answer the original user query. 
                    - DO NOT invent information.
                    - If tool call resulted in failure, analyze error and provide insight in the summary field.
                    """
                )

                EXECUTOR_FINAL_SYNTHESIZER = Instruction(
                    """
                    ### ROLE
                    You are a synthesis agent.  One or more tools have already been called and their
                    results are provided below.

                    ### YOUR TASKS
                    - Produce a clear, concise, self-contained answer for the step using only those
                    results.  
                
                    ### CONSTRAINTS
                    - Do NOT call any tools.
                     - If the tool results tell you to use internal knowledge, then use your internal knowledge to answer that part.
                    """
                )

            PLANNER = Instruction(
                """
                    ### ROLE
                    You are a planning agent. Your job is to decompose a user query into a minimal, ordered sequence of concrete steps that an executor can carry out one by one.

                    ### YOUR TASKS
                    1. Assess whether the query is clear and complete enough to plan.
                       - If it is too vague, ambiguous, or missing critical information, set needs_clarification=True and provide a single, specific follow-up question in clarification_question. Leave goal, plan_reasoning, and steps empty.
                       - If the query is clear, set needs_clarification=False and leave clarification_question empty.
                    2. Identify the ultimate goal of the user query.
                    3. Break it down into the smallest number of self-contained steps needed to achieve that goal.
                    4. Each step must describe exactly ONE actionable task (information retrieval, computation, lookup, etc.).
                    5. Steps must be ordered so that each one can use the results of all previous steps.
                    6. Set max_iterations to the maximum number of tool-call iterations the executor should be allowed per step (integer between 3 and 10). Use 3 for simple lookups or direct reasoning, and higher values for steps that may require multiple searches, retries, or chained tool calls.

                    ### CONSTRAINTS
                    - Do NOT include a step for summarizing, compiling, or presenting results — that is handled automatically after all steps complete.
                    - Do NOT add redundant or overlapping steps.
                    - Do NOT answer the query yourself.
                    - Ask for clarification only when genuinely necessary — not for minor ambiguities you can reasonably resolve yourself.
                    - If the conversation history shows the user declining a follow-up (e.g., "no", "not interested", "skip"), set steps to an empty list — no execution is needed.
                    - **Warm corpus rule**: if the conversation history shows that a web search was already performed and its results were indexed into a corpus, prefer a step that queries the existing corpus.
                    - **Executor capability rule**: if you are unsure whether a request can be fulfilled, still create the step and delegate it to the executor — the executor has access to tools and capabilities that you are not aware of, and may be able to handle it. Only skip a step if you are certain it is impossible or out of scope.
                    """
            )

            FINAL_COMPILER = Instruction(
                """
                ### ROLE
                You are a final answer compiler. A multi-step plan has been executed and each step's result is provided below.

                ### YOUR TASKS
                1. Combine all step results into a single, complete, and coherent answer to the original user query.
                2. Present the answer in a clear, well-structured format appropriate to the query.
                3. Suggest one context-aware follow-up to offer the user — something that naturally extends the topic just discussed.

                ### CONSTRAINTS
                - Use ONLY the provided step results. Do NOT add outside knowledge or speculation.
                - Do NOT mention steps, the planning process, or internal mechanics.
                - Do NOT call any tools.
                - The follow-up must be genuinely relevant to the answer — not generic filler. Leave follow_up_question empty if nothing natural comes to mind.
                - Phrase the follow-up as a conversational offer, not a bare question. Use a
                  suggestive tone such as "Would you like to know …?", "Do you want to explore …?",
                  or "Shall I tell you more about …?" — never a direct open question like "What is …?"
                """
            )

        class WebSearch:
            QUERY_OPTIMIZER = Instruction(
                """
                Transform the user's input into a high-density search engine query.
                - Output ONLY the optimized query. No conversational filler or quotes.
                - Use less than 10 high-impact keywords. Remove all stop words (the, a, is, how, to).
                - Focus on "Search Intent": rewrite questions as noun phrases or technical terms.
                - NEVER answer the query. Only provide the string used to find the answer.
                """
            )

        class RAG:
            """System instruction templates for the RAG agent's sub-tasks."""

            QUERY_REWRITER = Instruction(
                """
                Rewrite the user's query for optimal knowledge base retrieval.
                - Output ONLY the rewritten query. No explanation, or additional text.
                - Maximize recall: expand abbreviations, resolve pronouns, add key synonyms or related terms.
                - Maximize precision: remove ambiguity, make intent explicit.
                - Always rewrite as a declarative statement or keyword phrase, NEVER as a question.
                - Do NOT answer the query, or add the answer to the rewritten query. Rewrite the query for retrieval purposes.
                """
            )

            CONTEXT_SUMMARIZER = Instruction(
                """
                Summarize the retrieved context passages as they relate to the user's query.
                - Use ONLY information from the provided context. No outside knowledge.
                - Be BRIEF. Focus on relevance to the query, and the key facts or insights that would help answer it.
                - Do NOT exclude context passages that may be relevant, even if their relevance is not immediately clear.
                - Do NOT answer the query; only summarize relevant context.
                """
            )

            RESPONSE_GENERATOR = Instruction(
                """
                Answer the user's question using ONLY the provided context.
                - Be concise: 1-3 sentences unless more detail is explicitly requested.
                - No outside knowledge, no speculation beyond the context.
                - If context is insufficient, say "I don't know."
                - Context passages are labeled "Context 1", "Context 2", etc.
                - Cite sources inline using the format [1], [2], etc. matching the context number.
                - If the context is irrelevant or is not provided, DO NOT cite any of the context passages. Instead, say "I don't know."
                """
            )

            CITATION_CHECKER = Instruction(
                """
                You are a citation verifier. You will receive a user query, an LLM-generated answer, and numbered source context passages labeled [1], [2], etc.
                Your task is to verify that every factual claim in the answer is supported by the provided context.

                RULES:
                - "pass": all claims are supported.
                - "partial": some claims are supported, some are not.
                - "fail": majority of claims are unsupported or fabricated.
                - If no citations are present but a claim matches context content, mark it as supported and note the missing citation.
                - Do NOT answer the query. Only verify claims.

                You MUST respond with a single raw JSON object and nothing else.
                Do NOT use markdown. Do NOT use code fences. Do NOT add any explanation.
                Your entire response must start with { and end with }.

                Use this exact structure:
                {"verified": [{"claim": "<claim text>", "context_id": "<e.g. 1>", "supported": true}], "unsupported": [{"claim": "<claim text>", "reason": "<why it is not supported>"}], "verdict": "<pass|partial|fail>"}
                """
            )

    class CorpusManager:
        """Constants for RAG corpus management."""

        class MetadataKeys:
            """Dictionary keys used in chunk metadata throughout the RAG corpus."""

            DOCUMENT_ID = "document_id"
            DATA_CHUNK_ID = "data_chunk_id"
            START_WORD_INDEX = "start_word_index"
            DELETED = "deleted"

    class Tools:

        class InternalKnowledgeTool:
            NAME = "internal_knowledge"
            DESCRIPTION = """
                Uses the model's own training knowledge.
                Only use this when the step requires pure reasoning, math, coding, or a definition
                of a timeless concept AND no retrieval or search tool could provide better or more
                current information.
                DO NOT use if the answer depends on facts, events, data, or anything that exists
                outside the model — even partially. When in doubt, prefer a retrieval or search tool.
                """

        """Identifiers for registered agent tools."""

        class SupportedLLMs:
            GEMINI = "gemini"

        class Parameters:
            class Types:
                STRING = "string"
                INTEGER = "integer"
                FLOAT = "float"
                BOOLEAN = "boolean"
                LIST = "list"
                DICT = "dict"
                ARRAY = "array"
                OBJECT = "object"
                GENERIC = "generic"
                NUMBER = "number"

        class DatetimeTool:
            NAME = "get_current_datetime"
            DESCRIPTION = "Get the current date and time. Returns format: YYYY-MM-DD HH:MM:SS"

        class DaysSinceEpochTool:
            NAME = "get_days_since_epoch"
            DESCRIPTION = "Get the number of days since Unix epoch (January 1, 1970, 00:00:00 UTC). Returns an integer."

            class Parameters:
                class TargetDate:
                    NAME = "target_date"
                    DESCRIPTION = "The target date to count days from the Unix epoch to, in YYYY-MM-DD format."

        class RagTool:
            NAME = "rag_query"
            DESCRIPTION = """
                End-to-end RAG pipeline: rewrites the query for optimal retrieval, fetches relevant
                passages from a corpus, summarizes the context, and generates a grounded, cited answer.
                Use this when a complete, polished answer with context summarization and response
                generation is needed.
                It returns a dict with the generated response, retrieved context passages, and citation check results.
                """

            class Parameters:
                class Query:
                    NAME = "query"
                    DESCRIPTION = "The question or query to answer using the knowledge base."

                class CorpusName:
                    NAME = "corpus_name"
                    DESCRIPTION = "The name of the corpus to query. This must match one of the available corpus names."

                class SimilarityScoreThreshold:
                    NAME = "similarity_score_threshold"
                    DESCRIPTION = "Optional minimum similarity score for history matches. Range: [0.0, 10.0]. Default is 6.5."

                class SemanticSimilarityWeight:
                    NAME = "semantic_similarity_weight"
                    DESCRIPTION = "Optional weight for dense retrieval. Range: [0.0, 1.0]. Default is 0.8. (Note: semantic + lexical weights must sum to 1.0)"

                class LexicalSimilarityWeight:
                    NAME = "lexical_similarity_weight"
                    DESCRIPTION = "Optional weight for sparse retrieval. Range: [0.0, 1.0]. Default is 0.2. (Note: semantic + lexical weights must sum to 1.0)"

                class CrossEncoderTopK:
                    NAME = "top_k"
                    DESCRIPTION = "Optional number of passages to keep after re-ranking. Range: [1, 20]. Default is 5."

                class UseAgentHistory:
                    NAME = "use_agent_history"
                    DESCRIPTION = "Optional boolean whether to check agent history before fresh retrieval. Default is true."

        class AddDataToCorpusTool:
            NAME = "add_data_to_corpus"
            DESCRIPTION = """
                Adds new documents to an existing RAG corpus and re-indexes it so the new content
                becomes immediately searchable. Use this when new information needs to be ingested
                into the knowledge base.
                """

            class Parameters:
                class CorpusName:
                    NAME = "corpus_name"
                    DESCRIPTION = "The name of the corpus to add documents to. Must match one of the available corpus names."

                class Documents:
                    NAME = "documents"
                    DESCRIPTION = "A list of document strings to add to the corpus."

                class Metadatas:
                    NAME = "metadatas"
                    DESCRIPTION = "Optional list of metadata dictionaries, one per document, with arbitrary key-value pairs."

        class ClearCorpusTool:
            NAME = "clear_corpus"
            DESCRIPTION = """
                Removes all documents from the specified RAG corpus and resets its index.
                Use this to wipe a corpus before re-ingesting fresh content, or to clear cached
                results that were previously indexed into a corpus (e.g. from a prior web search, indexed document, ...).
                """

            class Parameters:
                class CorpusName:
                    NAME = "corpus_name"
                    DESCRIPTION = """
                        The name of the corpus to clear. Must match one of the available corpus names.
                        To clear previously indexed content, use the corpus name that the content was indexed into.
                        """

        class RelatedTopicsRetrieverTool:
            NAME = "related_topics_retriever"
            DESCRIPTION = """
                Retrieves and returns the top-k most relevant raw passages from a corpus for a given query.
                Performs ONLY retrieval — it does NOT summarize context or generate a final answer.
                Use this when you need direct access to the source passages themselves
                (e.g., for exploration, fact-checking, or feeding results into a downstream step).
                """

            class Parameters:
                class Query:
                    NAME = "query"
                    DESCRIPTION = "The rag-optimized statement or topic to find related passages for in the corpus."

                class CorpusName:
                    NAME = "corpus_name"
                    DESCRIPTION = "The name of the corpus to search. Must match one of the available corpus names."

                class TopK:
                    NAME = "top_k"
                    DESCRIPTION = "Optional number of top related items to retrieve. Default is 5 (choosing a higher number may provide more context but could also include less relevant passages). Range is 1 to 10."

        class WebSearchTool:
            NAME = "web_search"
            DESCRIPTION = """
                Fetches up-to-date web pages for a query, indexes them into a RAG corpus, and returns
                the top related raw passages (retrieval only — no answer generation). Use this when the
                question requires current or real-time information not available in the static knowledge base.
                The returned passages can be reasoned over by the downstream synthesizer.
                """

            class Parameters:
                class Query:
                    NAME = "query"
                    DESCRIPTION = "The search query to look up on the web."

                class CorpusName:
                    NAME = "corpus_name"
                    DESCRIPTION = "The name of the RAG corpus to index web results into and retrieve from. Must match one of the available corpus names."

                class MaxWebPages:
                    NAME = "max_web_pages"
                    DESCRIPTION = "Optional maximum number of web pages to fetch and index. Default is 5. The accepted range is 1 to 10. Setting this too high may lead to longer response times, while setting it too low may limit the information available for answering."

                class TopK:
                    NAME = "top_k"
                    DESCRIPTION = "Optional number of top related passages to return after retrieval and re-ranking. If omitted, defaults to the RAG handler's top_k setting."

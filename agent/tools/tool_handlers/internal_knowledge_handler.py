from base.constants import Constants
from base.data_classes import GenerativeModel, ToolResult


def use_internal_knowledge(query: str, generative_model: GenerativeModel) -> ToolResult:
    """Uses the generative model's own knowledge to answer a question directly.

    Args:
        query: The question or task to answer using internal knowledge.
        generative_model: The generative model to use for answering.

    Returns:
        ToolResult: .text contains the model's answer based on its training knowledge.
    """
    response = generative_model.generate(
        prompt=query,
        system_instructions=Constants.Instructions.InternalKnowledge.DIRECT_RESPONSE,
    )
    return ToolResult(text=response.text or "No answer could be generated.")

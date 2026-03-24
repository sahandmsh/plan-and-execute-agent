from base.data_classes import ToolResult


def use_internal_knowledge() -> ToolResult:
    """Uses the internal knowledge of the generative model to answer a question.

    Returns:
        ToolResult: .text is a message indicating that internal knowledge is being used to answer;
                    .data is a note that internal knowledge is sufficient to answer the question without external tools.
    """
    return ToolResult(
        text="Use your internal knowledge to answer as it is sufficient to answer the question without external tools.",
    )

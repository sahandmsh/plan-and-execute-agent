from base.constants import Constants
from base.data_classes import GenerativeModel, PlanningAgentResponse
from utils.chat_history_manager import ChatHistoryManager


def run_clarification(
    generative_model: GenerativeModel,
    chat_history_manager: ChatHistoryManager,
) -> tuple[str, PlanningAgentResponse]:
    """Collect the user's prompt and interactively resolve any clarifications.

    Responsibilities:
      1. Read the raw user prompt from stdin.
      2. Record it in the chat history.
      3. Loop with the planner model until it no longer needs clarification,
         asking the user follow-up questions and storing their answers.

    Args:
        generative_model:      The generative model used for planning.
        chat_history_manager:  Manages history persistence with a sliding window.

    Returns:
        A ``(user_prompt, plan)`` tuple where ``user_prompt`` is the original
        raw query and ``plan`` is the :class:`PlanningAgentResponse` with
        ``needs_clarification`` set to ``False``.
    """
    user_prompt = input("Prompt > ").strip()
    chat_history_manager.append("user", user_prompt)

    plan = _clarification_loop(
        user_prompt=user_prompt,
        generative_model=generative_model,
        chat_history_manager=chat_history_manager,
    )
    return user_prompt, plan


def _clarification_loop(
    user_prompt: str,
    generative_model: GenerativeModel,
    chat_history_manager: ChatHistoryManager,
) -> PlanningAgentResponse:
    """Resolve all clarification questions before planning begins.

    Loops until the planner model reports that it has enough information
    (``needs_clarification == False``). Each clarification question is shown
    to the user, their answer is recorded in history, and the next iteration
    re-queries the model with the updated context.

    Args:
        user_prompt:          The raw query submitted by the user.
        generative_model:     The generative model used for planning.
        chat_history_manager: Manages history persistence and the sliding window.

    Returns:
        The finalized :class:`PlanningAgentResponse` ready for execution.
    """
    while True:
        history_str = chat_history_manager.get_history_str()
        prompt = history_str if history_str else user_prompt
        response = generative_model.generate(
            prompt=prompt,
            system_instructions=Constants.Instructions.PlanAndExecuteAgent.PLANNER,
            response_schema=PlanningAgentResponse,
        )
        plan: PlanningAgentResponse = response.parsed

        if not plan.needs_clarification:
            break
        chat_history_manager.append("agent", plan.clarification_question)
        print(f"\n{plan.clarification_question}")
        answer = input("Your answer > ").strip()
        chat_history_manager.append("user", answer)
    return plan

from agent.executor_agent import ExecutorAgent
from base.constants import Constants
from base.data_classes import FinalSynthesisResponse, GenerativeModel, PlanningAgentResponse
from utils.chat_history_manager import ChatHistoryManager


def run_orchestrator(
    user_prompt: str,
    plan: PlanningAgentResponse,
    generative_model: GenerativeModel,
    executor: ExecutorAgent,
    chat_history_manager: ChatHistoryManager,
) -> str:
    """Execute a pre-clarified plan and synthesise a final answer.

    Clarification has already been handled by
    :func:`agent.interaction.clarification.run_clarification`
    before this function is called.

    Args:
        user_prompt:          The raw query submitted by the user.
        plan:                 The clarified :class:`PlanningAgentResponse` produced
                              by the clarification stage.
        generative_model:     The generative model used for final synthesis.
        executor:             The :class:`ExecutorAgent` that runs each plan step.
        chat_history_manager: Manages history persistence and the sliding window.
    """

    print(f"Goal: {plan.goal}")
    print(f"Plan reasoning: {plan.plan_reasoning}\n")
    step_results_context: list[str] = []
    for step in plan.steps:
        print(f"\n{'='*60}\nStep {step.step_number}: {step.step_description}\n{'-' * 60}")
        prior_context = "\n".join(
            f"Step {i + 1} result: {result}" for i, result in enumerate(step_results_context)
        )
        result = executor.execute_step(
            step_description=step.step_description,
            context=prior_context,
            max_iterations=plan.max_iterations,
        )
        step_results_context.append(result.answer)

    compiled_results = "\n".join(
        f"- {step.step_description}\n  Result: {ans}"
        for step, ans in zip(plan.steps, step_results_context)
    )
    final_synthesis_prompt = (
        f"Original user request: {user_prompt}\n\n"
        f"Goal: {plan.goal}\n\n"
        f"Step results:\n{compiled_results}"
    )
    final_response = generative_model.generate(
        prompt=final_synthesis_prompt,
        system_instructions=Constants.Instructions.PlanAndExecuteAgent.FINAL_COMPILER,
        response_schema=FinalSynthesisResponse,
    )
    final_answer = (
        final_response.parsed.final_answer if final_response.parsed else final_response.text or ""
    )
    if final_answer:
        print(f"All steps completed.\n{'='*60}\nFINAL ANSWER:\n{final_answer}\n{'='*60}\n")
    chat_history_manager.append("agent", final_answer)
    follow_up_question = final_response.parsed.follow_up_question if final_response.parsed else ""
    if follow_up_question:
        chat_history_manager.append("agent", follow_up_question)
        print(f"{follow_up_question}")
    return final_answer

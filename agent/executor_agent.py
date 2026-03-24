from typing import Callable
from agent.tools.tool_manager import ToolManager
from base.constants import Constants
from base.data_classes import (
    ExecutorStepResponse,
    GenerativeModel,
    InternalSynthesizerResponse,
    LLMResponse,
    StepResult,
    ToolCall,
    ToolResult,
)


class ExecutorAgent:
    def __init__(self, generative_model: GenerativeModel, tool_manager: ToolManager):
        self.generative_model = generative_model
        self.tool_manager = tool_manager

    def execute_step(
        self, step_description: str, max_iterations: int, context: str = ""
    ) -> StepResult:
        """Execute a single step of the plan using a ReAct loop.

        Args:
            step_description: Description of the step to execute, from the plan.
            max_iterations: Maximum number of ReAct iterations before giving up.
            context: Optional string with relevant information from previous steps.
        Returns:
            StepResult containing the final answer, tool call history, and any errors.
        """
        all_tool_schemas = self.tool_manager.get_all_tools_schema()
        observation = []
        tool_calls = []
        internal_synthesizer_history = []

        for iteration in range(1, max_iterations + 1):
            dispatcher_response: LLMResponse = self.generative_model.generate(
                prompt=self._prompt_generator(
                    step_description, context, observation, internal_synthesizer_history
                ),
                system_instructions=Constants.Instructions.PlanAndExecuteAgent.Executor.EXECUTOR_DISPATCHER,
                tools=all_tool_schemas,
            )

            if not dispatcher_response.function_calls:
                if not tool_calls:
                    break

            for function_call in dispatcher_response.function_calls or []:
                tool_name = function_call.name
                tool_args = dict(function_call.args) if function_call.args else {}

                print(
                    f"\n  [Iteration {iteration}] Calling tool: '{tool_name}' | args: {tool_args}"
                )

                if not self.tool_manager.has_tool(tool_name):
                    tool_response: ToolResult = ToolResult(
                        text=f"Tool {tool_name} not found in tool manager.",
                    )
                try:
                    if tool_args:
                        tool_response: ToolResult = self.tool_manager.use_tool(
                            tool_name, **tool_args
                        )
                    else:
                        tool_response: ToolResult = self.tool_manager.use_tool(tool_name)
                except Exception as e:
                    tool_response = ToolResult(
                        text=f"Tool {tool_name} execution raised an error",
                        data=f"Error: {str(e)}",
                    )
                tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=tool_response,
                    )
                )
                observation.append(
                    f"iteration: {iteration}\n[Tool: {tool_name}] args={tool_args}\nTool result: {tool_response.text}\n Tool data: {tool_response.data}"
                )

            internal_synthesizer_response: LLMResponse = self.generative_model.generate(
                prompt=self._prompt_generator(
                    step_description, context, observation, internal_synthesizer_history
                ),
                system_instructions=Constants.Instructions.PlanAndExecuteAgent.Executor.EXECUTOR_INTERNAL_SYNTHESIZER,
                response_schema=InternalSynthesizerResponse,
            )
            parsed: InternalSynthesizerResponse = internal_synthesizer_response.parsed
            internal_synthesizer_history.append(
                f"iteration: {iteration}\n[Internal Synthesizer] summary={parsed.summary}\nmissing_info={parsed.missing_info}\nearly_stop={parsed.early_stop}"
            )
            if parsed.early_stop:
                break

        # FINAL synthesizer to produce final answer
        final_synthesis_response: LLMResponse = self.generative_model.generate(
            prompt=self._prompt_generator(step_description, context, observation),
            system_instructions=Constants.Instructions.PlanAndExecuteAgent.Executor.EXECUTOR_FINAL_SYNTHESIZER,
            response_schema=ExecutorStepResponse,
        )
        parsed: ExecutorStepResponse = final_synthesis_response.parsed
        if parsed is None:
            return StepResult(
                step_description=step_description,
                answer=final_synthesis_response.text or "",
                tool_calls=tool_calls,
                iterations=len(tool_calls),
                error="Synthesis step returned unparsed response.",
            )
        return StepResult(
            step_description=step_description,
            answer=parsed.answer,
            tool_calls=tool_calls,
            iterations=len(tool_calls),
            success=True,
        )

    def _prompt_generator(
        self,
        step_description: str,
        context: str,
        observation: list[str] | None = None,
        internal_synthesizer_history: list[str] | None = None,
    ) -> str:
        """Generate the prompt for the executor's LLM calls, incorporating the step description,
        context from prior steps, observations from previous iterations, and internal synthesizer feedback.

        Args:
            step_description: Description of the step to execute, from the plan.
            context: Optional string with relevant information from previous steps.
            observation: Optional list of strings with observations from previous iterations.
            internal_synthesizer_history: Optional list of strings with internal synthesizer feedback.
        Returns:
            A string containing the generated prompt.
        """
        prompt = [f"Step Description: {step_description}\n"]
        if context:
            prompt.append(f"Context from prior steps:\n{context}\n")
        if observation:
            observations_str = "\n".join(observation)
            prompt.append(f"Observations from previous iterations:\n{observations_str}\n")
        if internal_synthesizer_history:
            internal_synthesizer_str = "\n".join(internal_synthesizer_history)
            prompt.append(f"Internal synthesizer feedback:\n{internal_synthesizer_str}\n")
        return "\n".join(prompt)

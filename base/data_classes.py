"""Centralized type definitions for the modular-planning-agent.

All ``@dataclass`` and Pydantic ``BaseModel`` types used across the codebase
are declared here so that every module can import from a single, well-known
location instead of pulling types from deep implementation files.
"""

from base.constants import Constants
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
from rag.rag_corpus_manager import RAGCorpusManager
from typing import Any, Callable, Optional


@dataclass
class CorpusInfo:
    """Structured representation of a RAG corpus for tool use.

    Attributes:
        name: Unique identifier for the corpus, used in tool parameters.
        description: Natural language description of the corpus content and purpose.
    """

    name: str
    corpus: RAGCorpusManager
    description: str


# ---------------------------------------------------------------------------
# LLM / model layer
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Standardised response from any language model."""

    text: str
    parsed: Optional[Any] = None
    function_calls: Optional[list[Any]] = None


@dataclass
class GenerativeModel:
    """A provider-agnostic wrapper for a generative language model.

    Decouples the rest of the codebase from any specific SDK or API client.
    Every agent and handler that needs text generation accepts a
    ``GenerativeModel`` instead of a raw ``Callable``, making the expected
    interface explicit and enabling easy swapping of the underlying model.

    Attributes:
        model_name: Human-readable identifier for the model (e.g. ``"gemini-2.5-flash"``).
                    Used for logging and debugging only — never passed to the model itself.
        generate:   Callable with the signature
                    ``(prompt, system_instructions="", response_schema=None, **kwargs) -> LLMResponse``.
                    Implemented and closed over by :meth:`~utils.model_loader.ModelLoader.load_gemini_generative_model`.
    """

    model_name: str
    generate: Callable[..., LLMResponse]


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Unified return type for every tool handler.

    Attributes:
        text: The LLM-facing string — concise, human-readable, ready to be
              injected directly into a prompt as an observation.
        data: Optional raw structured output for programmatic use by the caller
              (e.g. the full RAG response dict, an integer count, etc.).
              Never fed to the LLM directly.
    """

    text: str
    data: Optional[Any] = None


@dataclass(frozen=True)
class ParameterDetails:
    """Schema metadata for a single tool parameter."""

    type: Constants.Tools.Parameters.Types
    description: str
    required: bool = True
    items_type: Optional[str] = None  # element type for ARRAY parameters (e.g. "string", "object")


@dataclass(frozen=True)
class Tool:
    """Descriptor for a registered agent tool.

    Attributes:
        name:        Unique tool identifier used by the LLM when requesting calls.
        description: Natural-language description of what the tool does.
        parameters:  Mapping of parameter names to their :class:`ParameterDetails`.
        handler:     Callable that implements the tool; must return a
                     :class:`ToolResult`.  The ``text`` field is fed to the LLM;
                     ``data`` is for programmatic use only.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDetails]
    handler: Callable[..., ToolResult]


# ---------------------------------------------------------------------------
# Executor layer
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """Record of one tool invocation inside the executor loop."""

    tool_name: str
    arguments: dict
    result: ToolResult


@dataclass
class StepResult:
    """The resolved outcome for a single plan step."""

    step_description: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Gemini structured-output response schemas (Pydantic models)
# ---------------------------------------------------------------------------


class InternalSynthesizerResponse(BaseModel):
    """Structured response from the internal knowledge synthesizer."""

    summary: str = Field(description="A concise bulleted list of new facts gathered in this step.")
    missing_info: str = Field(
        description="If 'CONTINUE', list exactly what data point we are still hunting for."
    )
    early_stop: bool = Field(
        description="Set to true if the information is sufficient to consider this specific step 'Resolved', otherwise false."
    )


class PlanStep(BaseModel):
    """A single step inside a :class:`PlanningAgentResponse` plan."""

    step_number: int = Field(
        description="The sequential number of the step in the plan, starting from 1."
    )
    step_description: str = Field(
        description="A clear, self-contained description of what this step must accomplish, written so it can be handed directly to an executor."
    )


class PlanningAgentResponse(BaseModel):
    """Top-level structured response from the planning / orchestrator agent."""

    needs_clarification: bool = Field(
        description=(
            "Set to true if the user query is too vague, ambiguous, or incomplete to produce a reliable plan. "
            "Set to false if the query is clear enough to plan immediately."
        )
    )
    clarification_question: str = Field(
        description=(
            "If needs_clarification is true, a single, specific follow-up question to ask the user. "
            "If needs_clarification is false, leave this as an empty string."
        )
    )
    goal: str = Field(
        description="The ultimate end goal extracted from the user's query. Empty string if needs_clarification is true."
    )
    plan_reasoning: str = Field(
        description="The thought process used to break down the goal into the logical sequence of steps below. Empty string if needs_clarification is true."
    )
    steps: list[PlanStep] = Field(
        description="The ordered sequence of actionable steps to achieve the goal. Empty list if needs_clarification is true."
    )
    max_iterations: int = Field(
        description=(
            "The maximum number of tool-call iterations the executor should be allowed to perform "
            "for each step in this plan. Choose based on the expected complexity of the steps in the range [3, 10]."
        )
    )

    @field_validator("max_iterations")
    @classmethod
    def clamp_max_iterations(cls, v: int) -> int:
        return max(3, min(10, v))


class ExecutorStepResponse(BaseModel):
    """Structured response from the executor agent for a single plan step."""

    thought: str = Field(
        description="Reasoning about the current step and the answer derived from the available information."
    )
    answer: str = Field(
        description="The complete, concise answer for this step based on the tool results or direct reasoning."
    )


class FinalSynthesisResponse(BaseModel):
    """Structured response produced by the final answer synthesizer."""

    final_answer: str = Field(
        description="A complete, coherent answer to the original user query, synthesized from all step results."
    )
    follow_up_question: str = Field(
        description="A suggestive follow-up offer to the user, or an empty string if none."
    )

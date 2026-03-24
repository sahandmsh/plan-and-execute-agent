from base.constants import Constants
from base.data_classes import Tool, ToolResult
from google.genai import types as genai_types


class ToolManager:
    """Manages a collection of tools that an LLM agent can use, including their registration, schema generation for different LLMs, and execution.
    Args:
        generative_model_name (str): The name of the generative model (LLM)
        that the tool schemas should be generated for (e.g. "gemini").
    """

    def __init__(self, generative_model_name: Constants.Tools.SupportedLLMs):
        self.tools = {}
        self.generative_model_name = generative_model_name
        schema_generator = getattr(self, f"_generate_{self.generative_model_name}_schema", None)
        if not schema_generator:
            raise ValueError(
                f"Schema generator for model '{self.generative_model_name}' not implemented."
            )

    def add_tool(
        self,
        tool: Tool,
    ):
        """Add a new tool to the tool management system.

        Args:
            tool (Tool): Tool object containing name, description, parameters, and handler.
            The Tool dataclass should have:
            - name (str): Unique identifier for the tool
            - description (str): What the tool does
            - parameters (dict[str, ParameterDetails]): Parameter specifications
            - handler (Callable): Function that implements the tool

        Raises:
            ValueError: If a tool with the same name already exists.
            TypeError: If the callable is not actually callable.
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already exists. Use a different name.")
        if not callable(tool.handler):
            raise TypeError(f"Tool '{tool.name}' handler is not a valid function.")
        self.tools[tool.name] = tool

    def _generate_gemini_schema(self, tool_name: str) -> genai_types.Tool:
        """Generate a Gemini-native Tool object from a generic Tool.

        Args:
            tool_name (str): The name of the tool to generate schema for.

        Returns:
            genai_types.Tool: A Gemini SDK Tool object ready to pass to GenerateContentConfig.

        Raises:
            ValueError: If the tool does not exist.
        """
        tool: Tool = self.tools.get(tool_name, None)
        if not tool:
            raise ValueError(
                f"Tool '{tool_name}' does not exist. Use a different name or add the tool first."
            )

        properties = {}
        required = []

        # Gemini API does not support "FLOAT"; map it to "NUMBER"
        _type_map = {"float": "number", "list": "array", "dict": "object", "generic": "string"}
        if tool.parameters:
            for param_name, param_info in tool.parameters.items():
                # Gemini SDK accepts uppercase type strings (e.g. "STRING", "INTEGER", "NUMBER")
                mapped_type = _type_map.get(param_info.type.lower(), param_info.type).upper()
                items_schema = None
                if mapped_type == "ARRAY":
                    items_type = param_info.items_type or "string"
                    items_schema = genai_types.Schema(type=items_type.upper())
                properties[param_name] = genai_types.Schema(
                    type=mapped_type,
                    description=param_info.description,
                    items=items_schema,
                )
                if param_info.required:
                    required.append(param_name)

        function_declaration = genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=genai_types.Schema(
                type="OBJECT",
                properties=properties,
                required=required,
            ),
        )

        return genai_types.Tool(function_declarations=[function_declaration])

    def remove_tool(self, tool_name: str):
        """Removes a tool from the tool management system.
        Args:
            tool_name (str): The name of the tool to remove.
        Raises:
            ValueError: If the tool does not exist.
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        del self.tools[tool_name]

    def use_tool(self, tool_name: str, *tool_args, **tool_kwargs) -> ToolResult:
        """Call a registered tool and return its result.

        Args:
            tool_name (str): The name of the tool to use.
            *tool_args: Positional arguments forwarded to the handler.
            **tool_kwargs: Keyword arguments forwarded to the handler.

        Returns:
            ToolResult: Structured result; .text is the LLM-ready string,
                        .data holds the raw structured output for callers.

        Raises:
            ValueError: If the tool does not exist or has no handler.
            TypeError: If the handler does not return a ToolResult.
        """
        tool: Tool = self.tools.get(tool_name, None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")
        if not tool.handler:
            raise ValueError(f"Tool '{tool_name}' does not have a handler function.")
        result = tool.handler(*tool_args, **tool_kwargs)
        if not isinstance(result, ToolResult):
            raise TypeError(
                f"Handler for tool '{tool_name}' must return a ToolResult, "
                f"got {type(result).__name__}."
            )
        return result

    def get_all_tools_schema(self) -> list:
        """Get the schema for all registered tools in the format required by the specified LLM.
        Returns:
            list: A list of LLM-native tool objects (e.g. genai_types.Tool for Gemini).
        """
        schema_generator = getattr(self, f"_generate_{self.generative_model_name}_schema", None)
        return [schema_generator(tool_name) for tool_name in self.tools]

    def get_tool_schema(self, tool_name: str):
        """Get the schema for a specific tool in the format required by the specified LLM.
            tool_name (str): The name of the tool to get the schema for.
        Returns:
            The LLM-native tool object (e.g. genai_types.Tool for Gemini).
        Raises:
            ValueError: If the schema generator for the specified model is not implemented.
        """
        schema_generator = getattr(self, f"_generate_{self.generative_model_name}_schema", None)
        return schema_generator(tool_name)

    def list_tools(self) -> list[str]:
        """Get a list of all registered tool names.

        Returns:
            list[str]: List of tool names.
        """
        return list(self.tools.keys())

    def get_tools_summary(self) -> str:
        """Return a human-readable summary of all registered tools for use in planner prompts.

        Returns:
            str: A newline-separated list of 'tool_name: description' entries.
        """
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools.values())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_name (str): The name of the tool.

        Returns:
            bool: True if tool exists, False otherwise.
        """
        return tool_name in self.tools

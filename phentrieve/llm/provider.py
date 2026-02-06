"""
LLM provider abstraction using LiteLLM.

This module provides a unified interface to 100+ LLM providers through LiteLLM,
with runtime dependency detection and helpful error messages.

Supported providers (priority):
- GitHub Models (github/gpt-4o, github/gpt-4o-mini)
- Google Gemini (gemini/gemini-1.5-pro, gemini/gemini-1.5-flash)
- Anthropic (anthropic/claude-sonnet-4-20250514)
- OpenAI (openai/gpt-4o, gpt-4o)
- Ollama (ollama/llama3.1) - local, no auth required

Authentication is handled via environment variables:
- GITHUB_TOKEN for GitHub Models
- GEMINI_API_KEY for Google Gemini
- ANTHROPIC_API_KEY for Anthropic
- OPENAI_API_KEY for OpenAI
"""

import json
import logging
import os
import re
from typing import Any

from phentrieve.llm.types import LLMResponse, TokenUsage, ToolCall

logger = logging.getLogger(__name__)

# Tool definitions for Phentrieve functions
PHENTRIEVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_hpo_terms",
            "description": (
                "Search for HPO (Human Phenotype Ontology) terms matching a clinical "
                "phrase or symptom description. Returns the most relevant HPO terms "
                "with their IDs, names, and similarity scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Clinical phrase to search for, e.g., 'seizures', "
                            "'intellectual disability', 'abnormal gait'"
                        ),
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_clinical_text",
            "description": (
                "Process a section of clinical text through Phentrieve's full pipeline: "
                "semantic chunking, assertion/negation detection, and HPO term retrieval. "
                "Returns candidate HPO annotations with assertion status (affirmed/negated)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Clinical text section to process",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (en, de, es, fr, nl). Auto-detected if not specified.",
                        "default": "auto",
                    },
                },
                "required": ["text"],
            },
        },
    },
]


class LLMProviderError(Exception):
    """Exception raised for LLM provider errors."""

    pass


class LLMProvider:
    """
    Unified LLM provider using LiteLLM.

    This class provides a consistent interface for interacting with various
    LLM providers through LiteLLM's unified API.

    Attributes:
        model: The model identifier (e.g., "github/gpt-4o").
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "github/gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        """
        Initialize the LLM provider.

        Args:
            model: LiteLLM model string (e.g., "github/gpt-4o", "gemini/gemini-1.5-pro").
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.

        Raises:
            ImportError: If LiteLLM is not installed.
        """
        self._litellm = _get_litellm()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Extract provider from model string
        self.provider = self._extract_provider(model)

        # Validate authentication
        self._validate_auth()

    def _extract_provider(self, model: str) -> str:
        """Extract provider name from model string."""
        if "/" in model:
            return model.split("/")[0]
        # Default to OpenAI for models without prefix
        return "openai"

    def _validate_auth(self) -> None:
        """Validate that required authentication is available."""
        auth_vars = {
            "github": "GITHUB_TOKEN",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

        if self.provider in auth_vars:
            var_name = auth_vars[self.provider]
            if not os.environ.get(var_name):
                logger.warning(
                    "Environment variable %s not set for provider '%s'. "
                    "Requests may fail without authentication.",
                    var_name,
                    self.provider,
                )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions for function calling.
            tool_choice: How to handle tool calls ("auto", "none", or specific tool).

        Returns:
            LLMResponse containing the model's response.

        Raises:
            LLMProviderError: If the API request fails.
        """
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
            }

            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

            response = self._litellm.completion(**kwargs)

            # Parse tool calls from response
            tool_calls = []
            choice = response.choices[0]
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments),
                        )
                    )

            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                provider=self.provider,
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                raw_response=response.model_dump()
                if hasattr(response, "model_dump")
                else {},
            )

        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()

            # Provide helpful error messages for common issues

            # Check for invalid/unknown model errors
            model_not_found_patterns = [
                "model not found",
                "does not exist",
                "invalid model",
                "unknown model",
                "not supported",
            ]
            if any(pattern in error_lower for pattern in model_not_found_patterns) or (
                "models/" in error_lower and "not found" in error_lower
            ):
                raise LLMProviderError(
                    f"Model '{self.model}' not found or not supported. "
                    f"Check the model name and provider documentation."
                ) from e

            # Check for authentication errors
            if "authentication" in error_lower or "api key" in error_lower:
                raise LLMProviderError(
                    f"Authentication failed for {self.provider}. "
                    f"Please check your API key environment variable."
                ) from e

            if "rate limit" in error_lower:
                raise LLMProviderError(
                    f"Rate limit exceeded for {self.provider}. "
                    f"Please wait and try again."
                ) from e

            raise LLMProviderError(f"LLM request failed: {error_msg}") from e

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tool_executor: "ToolExecutor",
        max_iterations: int = 5,
    ) -> tuple[LLMResponse, list[ToolCall], TokenUsage]:
        """
        Send a completion request and handle tool calls iteratively.

        This method automatically executes tool calls and continues the
        conversation until the model stops requesting tools or max_iterations
        is reached.

        Args:
            messages: Initial conversation messages.
            tool_executor: Executor to run tool calls.
            max_iterations: Maximum number of tool call iterations.

        Returns:
            Tuple of (final_response, all_tool_calls, cumulative_token_usage).

        Raises:
            LLMProviderError: If the API request fails.
        """
        all_tool_calls: list[ToolCall] = []
        current_messages = messages.copy()
        cumulative_usage = TokenUsage()

        for iteration in range(max_iterations):
            response = self.complete(
                messages=current_messages,
                tools=PHENTRIEVE_TOOLS,
                tool_choice="auto",
            )

            # Accumulate token usage from this iteration
            cumulative_usage.add(response.usage)

            if not response.tool_calls:
                # No more tool calls, we're done
                return response, all_tool_calls, cumulative_usage

            # Execute each tool call and add results to messages
            current_messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for i, tc in enumerate(response.tool_calls)
                    ],
                }
            )

            for i, tc in enumerate(response.tool_calls):
                result = tool_executor.execute(tc.name, tc.arguments)
                # Create new ToolCall with result
                completed_tc = ToolCall(
                    name=tc.name,
                    arguments=tc.arguments,
                    result=result,
                )
                all_tool_calls.append(completed_tc)

                # Add tool result to messages
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": json.dumps(result)
                        if not isinstance(result, str)
                        else result,
                    }
                )

            tool_names = [tc.name for tc in response.tool_calls]
            logger.debug(
                "[LLM] Tool loop iteration %d of %d - executed %d call(s): %s",
                iteration + 1,
                max_iterations,
                len(response.tool_calls),
                ", ".join(tool_names),
            )

        # Max iterations reached
        logger.warning(
            "[LLM] Tool loop limit reached (%d iterations) - returning current response",
            max_iterations,
        )
        return response, all_tool_calls, cumulative_usage

    def supports_tools(self) -> bool:
        """Check if the current model supports native tool/function calling."""
        # Most modern models support tools, but some may not
        # LiteLLM handles this, but we can check explicitly
        no_tool_patterns = [
            r"ollama/(?!llama3)",  # Older Ollama models
            r"text-davinci",  # Legacy OpenAI
        ]
        for pattern in no_tool_patterns:
            if re.match(pattern, self.model):
                return False
        return True


class ToolExecutor:
    """
    Executes Phentrieve tools called by the LLM.

    This class provides the bridge between LLM tool calls and actual
    Phentrieve functionality.
    """

    def __init__(
        self,
        retriever: Any | None = None,
        text_processor: Any | None = None,
    ) -> None:
        """
        Initialize the tool executor.

        Args:
            retriever: Optional pre-initialized retriever instance.
            text_processor: Optional pre-initialized text processor.
        """
        self._retriever = retriever
        self._text_processor = text_processor

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool and return its result.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's result, typically a dict or list.

        Raises:
            ValueError: If the tool name is unknown.
        """
        if tool_name == "query_hpo_terms":
            return self._query_hpo_terms(
                query=arguments.get("query", ""),
                num_results=arguments.get("num_results", 5),
            )
        elif tool_name == "process_clinical_text":
            return self._process_clinical_text(
                text=arguments.get("text", ""),
                language=arguments.get("language", "auto"),
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _query_hpo_terms(
        self, query: str, num_results: int = 5
    ) -> list[dict[str, Any]]:
        """Execute the query_hpo_terms tool.

        Note: This requires a properly initialized retriever instance.
        For production use, set self._retriever before calling.
        """
        if self._retriever is None:
            # Attempt to create using query orchestrator for convenience
            try:
                from phentrieve.retrieval.query_orchestrator import orchestrate_query

                result = orchestrate_query(
                    query_text=query,
                    num_results=num_results,
                )
                # Extract results from orchestrator response
                if isinstance(result, dict) and "results" in result:
                    raw_results: list[dict[str, Any]] = result.get("results", [])
                    return [
                        {
                            "hpo_id": r.get("hpo_id", ""),
                            "term_name": r.get("term_name", ""),
                            "score": r.get("score", 0.0),
                            "definition": r.get("definition"),
                        }
                        for r in raw_results
                    ]
                return []
            except Exception as e:
                logger.warning(
                    "[TOOL] query_hpo_terms failed: %s - returning empty results", e
                )
                return []

        # Use provided retriever
        results = self._retriever.search(query, top_k=num_results)
        return [
            {
                "hpo_id": r.get("hpo_id", ""),
                "term_name": r.get("term_name", ""),
                "score": r.get("score", 0.0),
                "definition": r.get("definition"),
            }
            for r in results
        ]

    def _process_clinical_text(
        self,
        text: str,
        language: str = "auto",
    ) -> list[dict[str, Any]]:
        """Execute the process_clinical_text tool.

        Note: This requires a properly initialized text processor instance.
        For production use, set self._text_processor before calling.
        """
        if self._text_processor is None:
            # Attempt to use the text processing orchestrator
            try:
                from phentrieve.config import (
                    DEFAULT_MODEL,
                    get_sliding_window_punct_conj_cleaned_config,
                )
                from phentrieve.embeddings import load_embedding_model
                from phentrieve.retrieval.dense_retriever import DenseRetriever
                from phentrieve.text_processing.hpo_extraction_orchestrator import (
                    orchestrate_hpo_extraction,
                )
                from phentrieve.text_processing.pipeline import TextProcessingPipeline

                # Step 1: Load embedding model FIRST (needed for semantic chunking)
                model_name = DEFAULT_MODEL
                embedding_model = load_embedding_model(model_name)

                # Step 2: Process text into chunks with assertion detection
                # Use the full chunking pipeline config (includes conjunction splitting)
                # to properly separate phenotypes joined by "and", "or", etc.
                pipeline = TextProcessingPipeline(
                    language=language if language != "auto" else "en",
                    chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
                    assertion_config={"disable": False},
                    sbert_model_for_semantic_chunking=embedding_model,
                )
                chunks = pipeline.process(text)

                if not chunks:
                    return []

                # Step 2 (cont.): Extract chunk texts and assertion statuses
                chunk_texts = [c["text"] for c in chunks]
                assertion_statuses: list[str | None] = []
                for c in chunks:
                    status = c.get("status")
                    if status is None:
                        assertion_statuses.append("affirmed")
                    elif isinstance(status, str):
                        assertion_statuses.append(status)
                    else:
                        # AssertionStatus enum - get its value
                        assertion_statuses.append(status.value)

                # Step 3: Create retriever (using model loaded in Step 1)
                retriever = DenseRetriever.from_model_name(
                    model=embedding_model,
                    model_name=model_name,
                )

                if retriever is None:
                    logger.warning(
                        "[TOOL] process_clinical_text: retriever initialization failed"
                    )
                    return []

                # Step 4: Get HPO matches
                aggregated_results, _chunk_results = orchestrate_hpo_extraction(
                    text_chunks=chunk_texts,
                    retriever=retriever,
                    assertion_statuses=assertion_statuses,
                    language=language if language != "auto" else "en",
                )

                # Step 5: Format results for LLM consumption
                output = []
                for result in aggregated_results:
                    assertion = result.get("assertion_status")
                    if assertion is None:
                        assertion = "affirmed"

                    # Build evidence text from text_attributions
                    evidence_parts = []
                    for attr in result.get("text_attributions", []):
                        chunk_text = attr.get("chunk_text", "")
                        if chunk_text:
                            evidence_parts.append(chunk_text)
                    evidence_text = "; ".join(evidence_parts) if evidence_parts else ""

                    output.append(
                        {
                            "hpo_id": result.get("id", ""),
                            "term_name": result.get("name", ""),
                            "assertion": assertion,
                            "score": result.get("score", 0.0),
                            "evidence_text": evidence_text,
                        }
                    )
                return output

            except Exception as e:
                logger.warning(
                    "[TOOL] process_clinical_text failed: %s - returning empty results",
                    e,
                )
                return []

        # Use provided text processor
        results = self._text_processor.process(text)
        return [
            {
                "hpo_id": r.get("hpo_id", ""),
                "term_name": r.get("term_name", ""),
                "assertion": r.get("assertion", "affirmed"),
                "score": r.get("score", 0.0),
                "evidence_text": r.get("evidence_text"),
            }
            for r in results
        ]


def _get_litellm() -> Any:
    """
    Get the LiteLLM module with runtime detection.

    Returns:
        The litellm module.

    Raises:
        ImportError: If LiteLLM is not installed with helpful installation message.
    """
    try:
        import litellm

        # Suppress verbose logging from LiteLLM
        litellm.suppress_debug_info = True
        return litellm
    except ImportError as e:
        raise ImportError(
            "LiteLLM is not installed. The LLM annotation feature requires LiteLLM.\n\n"
            "Install with:\n"
            "    pip install litellm\n\n"
            "Or install Phentrieve with LLM support:\n"
            "    pip install phentrieve[llm]\n\n"
            "For specific providers, you may also need:\n"
            "    pip install anthropic        # For Anthropic/Claude\n"
            "    pip install openai           # For OpenAI\n"
            "    pip install google-generativeai  # For Google Gemini\n"
        ) from e


def get_available_models() -> dict[str, list[str]]:
    """
    Get a dictionary of available model presets by provider.

    Returns:
        Dict mapping provider names to lists of model identifiers.
    """
    return {
        "github": [
            "github/gpt-4o",
            "github/gpt-4o-mini",
        ],
        "gemini": [
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-exp",
            "gemini/gemini-1.5-pro",
            "gemini/gemini-1.5-flash",
        ],
        "anthropic": [
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-haiku-20241022",
        ],
        "openai": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "ollama": [
            "ollama/llama3.1",
            "ollama/llama3.1:70b",
            "ollama/mistral",
        ],
    }

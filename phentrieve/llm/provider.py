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

from phentrieve.llm.types import LLMResponse, TimingEvent, TokenUsage, ToolCall

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

        logger.debug(
            "[LLM] Starting tool loop (max %d iterations, %d initial messages)",
            max_iterations,
            len(current_messages),
        )

        import time as _time

        loop_t0 = _time.time()
        for iteration in range(max_iterations):
            logger.debug(
                "[LLM] Iteration %d/%d — sending request to %s",
                iteration + 1,
                max_iterations,
                self.model,
            )
            api_t0 = _time.time()
            response = self.complete(
                messages=current_messages,
                tools=PHENTRIEVE_TOOLS,
                tool_choice="auto",
            )
            api_elapsed = _time.time() - api_t0

            # Accumulate token usage from this iteration
            cumulative_usage.add(response.usage)
            cumulative_usage.llm_time_seconds += api_elapsed
            cumulative_usage.timing_events.append(
                TimingEvent(
                    label=f"LLM call #{iteration + 1} ({self.model})",
                    duration_seconds=api_elapsed,
                    category="llm",
                )
            )
            logger.debug(
                "[LLM] Iteration %d/%d — response in %.2fs: finish_reason=%s, "
                "tool_calls=%d, tokens=%s",
                iteration + 1,
                max_iterations,
                api_elapsed,
                response.finish_reason,
                len(response.tool_calls),
                response.usage,
            )

            if not response.tool_calls:
                # No more tool calls, we're done
                logger.debug(
                    "[LLM] Tool loop finished after %d iteration(s) in %.2fs, "
                    "total tokens: %d",
                    iteration + 1,
                    _time.time() - loop_t0,
                    cumulative_usage.total_tokens,
                )
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
                tool_t0 = _time.time()
                result = tool_executor.execute(tc.name, tc.arguments)
                tool_elapsed = _time.time() - tool_t0
                cumulative_usage.tool_time_seconds += tool_elapsed
                cumulative_usage.timing_events.append(
                    TimingEvent(
                        label=f"tool: {tc.name}({', '.join(f'{k}={v!r}' for k, v in tc.arguments.items())})"[
                            :120
                        ],
                        duration_seconds=tool_elapsed,
                        category="tool",
                    )
                )
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
        # Cached Phentrieve components (lazy-initialized on first use)
        self._cached_embedding_model: Any | None = None
        self._cached_retriever: Any | None = None
        self._cached_pipelines: dict[str, Any] = {}  # keyed by language

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
        import time as _time

        logger.debug(
            "[TOOL] Executing tool: %s (args: %s)", tool_name, list(arguments.keys())
        )
        _t0 = _time.time()

        if tool_name == "query_hpo_terms":
            result = self._query_hpo_terms(
                query=arguments.get("query", ""),
                num_results=arguments.get("num_results", 5),
            )
        elif tool_name == "process_clinical_text":
            result = self._process_clinical_text(
                text=arguments.get("text", ""),
                language=arguments.get("language", "auto"),
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        result_count = len(result) if isinstance(result, list) else "n/a"
        logger.debug(
            "[TOOL] Tool %s completed in %.2fs — %s results",
            tool_name,
            _time.time() - _t0,
            result_count,
        )
        return result

    def _query_hpo_terms(
        self, query: str, num_results: int = 5
    ) -> list[dict[str, Any]]:
        """Execute the query_hpo_terms tool using the cached retriever."""
        try:
            # Ensure retriever is initialized (cached after first call)
            _embedding_model, _pipeline, retriever = self._get_phentrieve_components(
                "en"
            )
            if retriever is None:
                return []

            raw = retriever.query(query, n_results=num_results)

            # Format ChromaDB-style results into clean dicts
            output: list[dict[str, Any]] = []
            metadatas = raw.get("metadatas", [[]])[0]
            similarities = (
                raw.get("similarities", [[]])[0] if raw.get("similarities") else []
            )
            for i, meta in enumerate(metadatas):
                score = similarities[i] if i < len(similarities) else 0.0
                output.append(
                    {
                        "hpo_id": meta.get("hpo_id", ""),
                        "term_name": meta.get("label", ""),
                        "score": score,
                    }
                )
            return output
        except Exception as e:
            logger.warning(
                "[TOOL] query_hpo_terms failed: %s - returning empty results", e
            )
            return []

    def _get_phentrieve_components(self, language: str) -> tuple[Any, Any, Any]:
        """Get or create cached Phentrieve pipeline components.

        Returns:
            Tuple of (embedding_model, pipeline, retriever).
        """
        from phentrieve.config import (
            DEFAULT_MODEL,
            get_sliding_window_punct_conj_cleaned_config,
        )
        from phentrieve.embeddings import load_embedding_model
        from phentrieve.retrieval.dense_retriever import DenseRetriever
        from phentrieve.text_processing.pipeline import TextProcessingPipeline

        lang_key = language if language != "auto" else "en"

        # Embedding model (cached globally by load_embedding_model)
        if self._cached_embedding_model is None:
            model_name = DEFAULT_MODEL
            logger.debug("[TOOL] Loading embedding model: %s (first use)", model_name)
            import time as _time

            _t0 = _time.time()
            self._cached_embedding_model = load_embedding_model(model_name)
            logger.debug("[TOOL] Embedding model loaded in %.2fs", _time.time() - _t0)
        else:
            logger.debug("[TOOL] Embedding model: using cached instance")

        # Text processing pipeline (cached per language)
        if lang_key not in self._cached_pipelines:
            logger.debug(
                "[TOOL] Creating TextProcessingPipeline for language '%s' (first use)",
                lang_key,
            )
            import time as _time

            _t0 = _time.time()
            self._cached_pipelines[lang_key] = TextProcessingPipeline(
                language=lang_key,
                chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
                assertion_config={"disable": False},
                sbert_model_for_semantic_chunking=self._cached_embedding_model,
            )
            logger.debug(
                "[TOOL] TextProcessingPipeline (%s) created in %.2fs",
                lang_key,
                _time.time() - _t0,
            )
        else:
            logger.debug(
                "[TOOL] TextProcessingPipeline (%s): using cached instance", lang_key
            )

        # Dense retriever (cached, reuses ChromaDB connection)
        if self._cached_retriever is None:
            model_name = DEFAULT_MODEL
            logger.debug(
                "[TOOL] Creating DenseRetriever for model '%s' (first use)", model_name
            )
            import time as _time

            _t0 = _time.time()
            self._cached_retriever = DenseRetriever.from_model_name(
                model=self._cached_embedding_model,
                model_name=model_name,
            )
            logger.debug("[TOOL] DenseRetriever created in %.2fs", _time.time() - _t0)
        else:
            logger.debug("[TOOL] DenseRetriever: using cached instance")

        return (
            self._cached_embedding_model,
            self._cached_pipelines[lang_key],
            self._cached_retriever,
        )

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
            try:
                from phentrieve.text_processing.hpo_extraction_orchestrator import (
                    orchestrate_hpo_extraction,
                )

                _embedding_model, pipeline, retriever = self._get_phentrieve_components(
                    language
                )

                if retriever is None:
                    logger.warning(
                        "[TOOL] process_clinical_text: retriever initialization failed"
                    )
                    return []

                # Process text into chunks with assertion detection
                chunks = pipeline.process(text)

                if not chunks:
                    return []

                chunk_texts = [c["text"] for c in chunks]
                assertion_statuses: list[str | None] = []
                for c in chunks:
                    status = c.get("status")
                    if status is None:
                        assertion_statuses.append("affirmed")
                    elif isinstance(status, str):
                        assertion_statuses.append(status)
                    else:
                        assertion_statuses.append(status.value)

                # Get HPO matches
                aggregated_results, _chunk_results = orchestrate_hpo_extraction(
                    text_chunks=chunk_texts,
                    retriever=retriever,
                    assertion_statuses=assertion_statuses,
                    language=language if language != "auto" else "en",
                )

                # Format results for LLM consumption
                output = []
                for result in aggregated_results:
                    assertion = result.get("assertion_status")
                    if assertion is None:
                        assertion = "affirmed"

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

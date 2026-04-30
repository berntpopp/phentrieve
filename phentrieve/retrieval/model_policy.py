"""Shared retrieval model allowlist and trust policy."""

from dataclasses import dataclass

from phentrieve.config import BENCHMARK_MODELS, DEFAULT_MODEL


@dataclass(frozen=True)
class RetrievalModelPolicy:
    """Resolved retrieval model settings owned by the server."""

    model_name: str
    trust_remote_code: bool


def allowed_retrieval_model_names() -> tuple[str, ...]:
    """Return retrieval models accepted by public API request fields."""
    return tuple(dict.fromkeys([DEFAULT_MODEL, *BENCHMARK_MODELS]))


def resolve_retrieval_model_policy(model_name: str | None) -> RetrievalModelPolicy:
    """Resolve a request model name to a server-owned retrieval model policy."""
    resolved_model_name = model_name or DEFAULT_MODEL
    allowed_models = allowed_retrieval_model_names()
    if resolved_model_name not in allowed_models:
        allowed = ", ".join(allowed_models)
        raise ValueError(
            f"Unsupported retrieval model: {resolved_model_name}. "
            f"Allowed values: {allowed}."
        )

    return RetrievalModelPolicy(
        model_name=resolved_model_name,
        trust_remote_code=resolved_model_name == DEFAULT_MODEL,
    )

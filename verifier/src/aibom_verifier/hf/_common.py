from collections.abc import Generator
from contextlib import contextmanager

from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from aibom_verifier.types import CompareStartError


def _http_status_code(exc: HfHubHTTPError) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


@contextmanager
def hub_errors(repo_id: str, context: str) -> Generator[None]:
    """Translate all Hub exceptions into CompareStartError.

    GatedRepoError subclasses RepositoryNotFoundError in some hub
    versions, so it must be caught first.
    """
    try:
        yield
    except GatedRepoError as exc:
        raise CompareStartError(
            "gated_unauthenticated",
            f"Repo '{repo_id}' is gated; authenticate with a token that has access.",
        ) from exc
    except RepositoryNotFoundError as exc:
        raise CompareStartError(
            "repo_not_found",
            f"Repo '{repo_id}' was not found ({context}).",
        ) from exc
    except EntryNotFoundError as exc:
        raise CompareStartError(
            "resolve_failed",
            f"Entry not found for '{repo_id}' ({context}): {exc}",
        ) from exc
    except HfHubHTTPError as exc:
        code = _http_status_code(exc)
        if code in (401, 403):
            raise CompareStartError(
                "gated_unauthenticated",
                f"Repo '{repo_id}' requires authentication (HTTP {code}).",
            ) from exc
        if code == 404:
            raise CompareStartError(
                "repo_not_found",
                f"Repo '{repo_id}' was not found (HTTP 404).",
            ) from exc
        raise CompareStartError(
            "resolve_failed",
            f"Failed to access '{repo_id}' ({context}): {exc}",
        ) from exc
    except CompareStartError:
        raise
    except Exception as exc:
        raise CompareStartError(
            "resolve_failed",
            f"Failed to access '{repo_id}' ({context}): {exc}",
        ) from exc

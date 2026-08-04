from aibom_verifier.types import CompareStartError


class NotSafetensorsError(ValueError):
    """Raised when a repo has no safetensors weights."""

    def __init__(self) -> None:
        super().__init__("not_safetensors")


__all__ = ["CompareStartError", "NotSafetensorsError"]

"""Where a single detection test runs.

``ExecutionBackend`` is an alias of :class:`~aibom_verifier.slots.worker.Worker`
(same ``run(node_id, inputs, store)`` contract). Prefer this name at the
orchestrator boundary.

**Local** backends use the ``store`` argument as-is.

**Compose / SSH** backends ignore the passed store object and rebuild from env
(``AIBOM_STORE``, ``AIBOM_PG_DSN``, MinIO vars, …) and/or job ``store_config``
(Compose Redis payload).

**OpenShift AI / MetaCentrum** are upgrade paths on this same interface only —
not M1 demos.
"""

from __future__ import annotations

from aibom_verifier.slots.worker import Worker

ExecutionBackend = Worker

__all__ = ["ExecutionBackend"]

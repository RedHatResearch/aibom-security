"""Default node registry (leaf module — no backend / orchestrator imports)."""

from __future__ import annotations

from aibom_verifier.nodes.arch_hash_gate import arch_hash_gate_node
from aibom_verifier.nodes.block0_shapes import block0_shapes_node
from aibom_verifier.nodes.block0_values import block0_values_node
from aibom_verifier.nodes.resolve_refs import resolve_refs_node
from aibom_verifier.nodes.support_classify import support_classify_node
from aibom_verifier.slots.worker import NodeFn

DEFAULT_REGISTRY: dict[str, NodeFn] = {
    "resolve_refs": resolve_refs_node,
    "support_classify": support_classify_node,
    "arch_hash": arch_hash_gate_node,
    "block0_shapes": block0_shapes_node,
    "block0_values": block0_values_node,
}

__all__ = ["DEFAULT_REGISTRY"]

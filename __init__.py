"""Castcut post-render checks for ComfyUI (see README.md). The nodes live in castcut_nodes.py."""

from .castcut_nodes import CASTCUT_VERSION, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = CASTCUT_VERSION

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

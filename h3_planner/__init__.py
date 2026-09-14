"""H3 Planner — multi-segment timeline orchestration for MiniMax H3."""

from . import (nodes_beat, nodes_cast, nodes_plan, nodes_prompt,
               nodes_run, nodes_slice,
               nodes_story)

NODE_CLASS_MAPPINGS = {}
for _mod in (nodes_plan, nodes_cast, nodes_beat, nodes_prompt,
             nodes_slice, nodes_story,
             nodes_run):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
for _mod in (nodes_plan, nodes_cast, nodes_beat, nodes_prompt,
             nodes_slice, nodes_story,
             nodes_run):
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

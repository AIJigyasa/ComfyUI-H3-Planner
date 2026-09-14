"""ComfyUI-H3-Planner — multi-segment timeline orchestration for MiniMax H3.

Plans a long video as segments, hands the sampler one segment per queue run,
stores every result in a timeline-addressed vault, and stitches the result
back to the planned durations.

Sampling is deliberately outside this pack: it emits plain primitives that
wire straight into MiniMaxH3ReferenceToVideo and takes frames back afterwards.
"""

from .h3_planner import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .h3_planner import routes

routes.register()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("[H3Planner] loaded %d nodes" % len(NODE_CLASS_MAPPINGS))

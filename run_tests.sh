#!/bin/sh
# Every suite. The end-to-end one is the important one: py_compile cannot see
# an undefined name inside a function body, so only calling the nodes for real
# catches a NameError before ComfyUI does.
set -e
# Browser modules first: a SyntaxError here removes a node's whole UI,
# and no Python test can see it.
sh tests/check_js.sh

for t in tests/test_core.py tests/test_prompt.py tests/test_end_to_end.py tests/test_io.py tests/test_story.py tests/test_slice.py tests/test_refine.py tests/test_beatmap.py tests/test_vocals.py tests/test_bridge.py; do
  printf '%-28s ' "$t"
  python "$t" > /dev/null && echo "pass"
done

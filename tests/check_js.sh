#!/bin/sh
# Parse every browser module the way the browser will.
#
# `node --check foo.js` parses as a SCRIPT, not a module, and bailed on the
# first `import` line — so it silently passed a file with two `const note`
# declarations in one function. That is a SyntaxError the browser refuses,
# which meant the H3 Timeline lost its card strip entirely and fell back to a
# raw JSON textarea. Copying to .mjs makes node parse it as a module.
set -e
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
status=0
for f in js/*.js; do
  name=$(basename "$f" .js)
  sed 's#^import .*from .*$#// import stripped for the parse check#' "$f" \
    > "$tmp/$name.mjs"
  if node --check "$tmp/$name.mjs" 2>"$tmp/err"; then
    printf '%-28s parses as a module\n' "$f"
  else
    printf '%-28s FAILED\n' "$f"
    sed -n '1,6p' "$tmp/err"
    status=1
  fi
done
exit $status

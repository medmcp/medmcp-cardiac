#!/usr/bin/env bash
# Vendor the CineMA model code this stack runs, from a pinned upstream commit.
#
# Usage:
#   ./scripts/vendor-cinema.sh            # clone upstream at CINEMA_REV
#   CINEMA_SRC=/path/to/CineMA ./scripts/vendor-cinema.sh   # reuse a checkout
#
# Why vendor rather than depend on the `cinema` package: upstream is not on PyPI,
# pins every dependency to an exact version (numpy 1.26.4 has no Python 3.13 wheel,
# which alone breaks CI's 3.13 leg), and drags in training-only packages (wandb,
# plotly, hydra, pandas, scikit-learn, MONAI, SimpleITK) that inference never
# touches. The model itself is six files. They are copied verbatim -- only the
# import paths change -- so a checkpoint loads into exactly the architecture it
# was trained with, and re-running this script against a newer CINEMA_REV is the
# whole upgrade path. The six timm layers those files import live in `_timm.py`,
# which is hand-maintained and not touched here.
#
# Everything under src/medmcp_cardiac/_cinema/ except __init__.py and _timm.py is
# overwritten by this script; do not edit those files by hand.

set -euo pipefail

CINEMA_REV="${CINEMA_REV:-c10daa1d93f0ea28d8b9ad9206b0f673d25805c1}"
CINEMA_REPO="https://github.com/mathpluscode/CineMA"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$repo_root/src/medmcp_cardiac/_cinema"

workdir=""
if [[ -z "${CINEMA_SRC:-}" ]]; then
    workdir="$(mktemp -d)"
    trap 'rm -rf "$workdir"' EXIT
    git clone -q "$CINEMA_REPO" "$workdir/CineMA"
    git -C "$workdir/CineMA" checkout -q "$CINEMA_REV"
    CINEMA_SRC="$workdir/CineMA"
fi

actual_rev="$(git -C "$CINEMA_SRC" rev-parse HEAD)"
if [[ "$actual_rev" != "$CINEMA_REV" ]]; then
    echo "error: $CINEMA_SRC is at $actual_rev, expected $CINEMA_REV" >&2
    exit 1
fi

# upstream path -> vendored module name
declare -A files=(
    ["cinema/conv.py"]="conv.py"
    ["cinema/convvit.py"]="convvit.py"
    ["cinema/vit.py"]="vit.py"
    ["cinema/rotary.py"]="rotary.py"
    ["cinema/log.py"]="log.py"
    ["cinema/transform.py"]="transform.py"
    ["cinema/segmentation/convunetr.py"]="convunetr.py"
)

mkdir -p "$dest"
for src in "${!files[@]}"; do
    out="$dest/${files[$src]}"
    sed \
        -e 's/^from cinema\.segmentation\.convunetr import /from medmcp_cardiac._cinema.convunetr import /' \
        -e 's/^from cinema\./from medmcp_cardiac._cinema./' \
        -e 's/^from timm\.layers import /from medmcp_cardiac._cinema._timm import /' \
        -e 's/^from timm\.models\.vision_transformer import /from medmcp_cardiac._cinema._timm import /' \
        "$CINEMA_SRC/$src" > "$out"
    echo "  vendored: $src -> src/medmcp_cardiac/_cinema/${files[$src]}"
done
cp "$CINEMA_SRC/LICENSE" "$dest/LICENSE"

# Nothing may still reach for the upstream package names.
if grep -nE '^(from|import) (cinema|timm)\b' "$dest"/*.py; then
    echo "error: an upstream import survived the rewrite (see above)" >&2
    exit 1
fi

{
    echo "# Written by scripts/vendor-cinema.sh -- do not edit."
    echo "repo: $CINEMA_REPO"
    echo "rev: $CINEMA_REV"
    echo "files:"
    for src in "${!files[@]}"; do
        echo "  $src -> ${files[$src]}"
    done | sort
} > "$dest/UPSTREAM"

echo "Done. Vendored CineMA @ ${CINEMA_REV:0:12} into src/medmcp_cardiac/_cinema/."

#!/bin/bash

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/testprojects"

# The last commit before 2026-05-01 UTC on each default branch, so the three clones describe one
# calendar moment. Resolved with
#   git rev-list -1 --first-parent --before=2026-05-01T00:00:00Z <branch>
# and pinned by SHA, since the branch ref can move and a date alone is not a pin.
PROJECTS=(
    "networkx|https://github.com/networkx/networkx.git|24861327ed3bc9a45c28b38f6e4b2ec6c59e678f"  # 3.6.1+121,  2026-04-29
    "gson|https://github.com/google/gson.git|8260eddffe413c132b3a38e68446da6bd083220d"            # 2.14.0+3,   2026-04-29
    "axios|https://github.com/axios/axios.git|ad68e1a484b50086af427f767bbd7d6e3aab7ac3"           # v1.15.2+37, 2026-04-30
)

mkdir -p "$DIR"

for entry in "${PROJECTS[@]}"; do
    IFS='|' read -r name repo commit <<< "$entry"
    path="$DIR/$name"
    if [ ! -d "$path/.git" ]; then
        echo "[clone] $name"
        git init -q "$path"
        git -C "$path" fetch -q --depth 1 "$repo" "$commit"
        git -C "$path" checkout -q FETCH_HEAD
    elif [ "$(git -C "$path" rev-parse HEAD)" = "$commit" ]; then
        echo "[ok]    $name"
    else
        echo "[!]     $name is not at pinned ${commit:0:9}, numbers will not match"
    fi

    # vitest is in axios's devDependencies, which cloning does not bring.
    if [ "$name" = "axios" ] && [ ! -d "$path/node_modules" ]; then
        echo "[deps]  $name: npm ci"
        (cd "$path" && npm ci --no-audit --no-fund >/dev/null)
    fi
done

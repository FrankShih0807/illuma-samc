#!/usr/bin/env bash
# publish.sh — Export a clean copy of illuma-samc to the public repo.
#
# Usage:
#   ./publish.sh              # dry-run: build export, show what would be pushed
#   ./publish.sh --push       # build export and push to public repo
#
# Prerequisites:
#   1. Create a public repo: gh repo create FrankShih0807/illuma-samc --public
#   2. Run this script from the private repo root.

set -euo pipefail

PUBLIC_REPO="git@github.com:FrankShih0807/illuma-samc.git"
EXPORT_DIR="/tmp/illuma-samc-publish"
BRANCH="main"

# Files/dirs to strip from the public release
EXCLUDE=(
    # Agent coordination
    "CLAUDE.md"
    "STATUS.md"
    "WORKLOG.md"
    "TODO.md"
    "BLOCKED.md"
    "FRICTION.md"
    "CROSSPROJECT.md"

    # Internal experiments and reference
    "ablation/"
    "reference/"
    "configs/"
    "compare_results.py"

    # Internal scripts
    "publish.sh"
    "scripts/"
)

echo "=== illuma-samc publish ==="
echo "Source:  $(pwd)"
echo "Target:  $PUBLIC_REPO"
echo ""

# Step 1: Clean export from git (respects .gitignore)
rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"
git archive HEAD | tar -x -C "$EXPORT_DIR"
echo "[1/4] Exported git HEAD to $EXPORT_DIR"

# Step 2: Remove internal files
for item in "${EXCLUDE[@]}"; do
    target="$EXPORT_DIR/$item"
    if [ -e "$target" ]; then
        rm -rf "$target"
        echo "  removed: $item"
    fi
done
echo "[2/4] Stripped internal files"

# Step 3: Show what will be published
echo ""
echo "Files to publish:"
(cd "$EXPORT_DIR" && find . -type f | sort | sed 's|^\./||')
echo ""
file_count=$(cd "$EXPORT_DIR" && find . -type f | wc -l | tr -d ' ')
echo "Total: $file_count files"

# Step 4: Push to public repo (if --push)
if [[ "${1:-}" == "--push" ]]; then
    echo ""
    echo "[3/4] Preparing git commit..."
    cd "$EXPORT_DIR"

    git init -b "$BRANCH" --quiet
    git add -A

    # Get the private repo's latest commit info for reference
    PRIVATE_SHA=$(cd - > /dev/null && git rev-parse --short HEAD)

    git commit -m "Release from private repo ($PRIVATE_SHA)" \
        --author="Frank Shih <fshih37@gmail.com>" --quiet

    echo "[4/4] Pushing to $PUBLIC_REPO..."
    git remote add public "$PUBLIC_REPO"
    git push public "$BRANCH" --force

    echo ""
    echo "Published to $PUBLIC_REPO"
else
    echo ""
    echo "Dry run complete. Run './publish.sh --push' to publish."
fi

# Cleanup
rm -rf "$EXPORT_DIR"

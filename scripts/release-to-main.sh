#!/bin/bash
#
# Curated merge from dev → main.
#
# Strips internal-only files (handoff docs, agent hooks) so main stays
# clean for the public. Creates a squash-merge commit on main.
#
# Usage:
#   ./scripts/release-to-main.sh              # interactive
#   ./scripts/release-to-main.sh "Release v2" # with message
#

set -euo pipefail

# Files that exist on dev but should NOT appear on main
INTERNAL_FILES=(
    "docs/HANDOFF-SCAV.md"
    "docs/HANDOFF-SCAVIEFAE.md"
    ".claude/hooks/check-handoffs.sh"
    ".claude/hooks/scan-gh-secrets.sh"
    ".claude/settings.json"
)

# Sanity checks
CURRENT=$(git branch --show-current)
if [ "$CURRENT" != "dev" ]; then
    echo "Error: must be on dev branch (currently on $CURRENT)"
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working tree not clean. Commit or stash first."
    exit 1
fi

# Get commit message
MSG="${1:-}"
if [ -z "$MSG" ]; then
    echo "Commits on dev since last main merge:"
    echo "---"
    git log main..dev --oneline
    echo "---"
    read -p "Release message: " MSG
    if [ -z "$MSG" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Merging dev → main (squash), stripping internal files..."
echo ""

# Switch to main, squash-merge dev
git checkout main
git merge --squash dev

# Remove internal files from the staged merge
for f in "${INTERNAL_FILES[@]}"; do
    if git diff --cached --name-only | grep -q "^${f}$"; then
        git reset HEAD -- "$f" >/dev/null 2>&1 || true
        git checkout -- "$f" 2>/dev/null || true
    fi
    # If the file was added by the merge, unstage and remove
    if [ -f "$f" ]; then
        git rm -f --cached "$f" >/dev/null 2>&1 || true
        rm -f "$f"
    fi
done

# Commit
git commit -m "$MSG

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

echo ""
echo "Done. Review with: git log --oneline -3"
echo "Push with: git push origin main"
echo ""

# Switch back to dev
git checkout dev

#!/bin/bash
#
# Post-execution hook: after git pull from main, check handoff docs for updates.
#
# Fires after any Bash command containing "git pull". Reads the two handoff
# docs and surfaces any recent changes so the agent can review and respond.
#

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only trigger on git pull commands
if ! echo "$COMMAND" | grep -qE 'git pull'; then
    exit 0
fi

SCAV_HANDOFF="docs/HANDOFF-SCAV.md"
SCAVIEFAE_HANDOFF="docs/HANDOFF-SCAVIEFAE.md"

# Check if either handoff doc changed in the pull
CHANGED=""

if git diff HEAD@{1}..HEAD --name-only 2>/dev/null | grep -q "$SCAV_HANDOFF"; then
    CHANGED="$CHANGED $SCAV_HANDOFF"
fi

if git diff HEAD@{1}..HEAD --name-only 2>/dev/null | grep -q "$SCAVIEFAE_HANDOFF"; then
    CHANGED="$CHANGED $SCAVIEFAE_HANDOFF"
fi

if [ -n "$CHANGED" ]; then
    # Show what changed in the handoff docs
    DIFF=$(git diff HEAD@{1}..HEAD -- $CHANGED 2>/dev/null | head -200)

    jq -n --arg files "$CHANGED" --arg diff "$DIFF" '{
        decision: "allow",
        reason: ("Handoff docs changed in pull:" + $files + "\n\nReview the changes and update your handoff doc if you have pending items to communicate (planning changes, questions, blockers — NOT status updates).\n\nDiff:\n" + $diff)
    }'
else
    jq -n '{
        decision: "allow",
        reason: "No handoff doc changes in this pull. Check if you have pending items to communicate to the other agent (planning changes, questions, blockers)."
    }'
fi

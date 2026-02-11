#!/bin/bash
#
# Pre-execution hook: block gh commands that contain secrets.
# Mirrors the patterns in scripts/pre-commit for consistency.
#

# Read hook input from stdin
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only scan gh issue/pr commands that post content
if ! echo "$COMMAND" | grep -qE 'gh (issue create|issue comment|pr create|pr comment|pr review)'; then
    exit 0
fi

# Known test keys that are safe (from eth-account docs / test fixtures)
ALLOWLIST="4c0883a69102937d6231471b5dbb6204fe512961708279f3a3e6d8b4f8e2c7e1"

FOUND=0
REASON=""

# Pattern 1: 0x + 64 hex chars (Ethereum private key format)
MATCHES=$(echo "$COMMAND" | grep -oE '0x[0-9a-fA-F]{64}' || true)
if [ -n "$MATCHES" ]; then
    while IFS= read -r match; do
        hex=$(echo "$match" | grep -oE '[0-9a-fA-F]{64}')
        if [ "$hex" != "$ALLOWLIST" ]; then
            FOUND=1
            REASON="Potential Ethereum private key (0x + 64 hex chars)"
            break
        fi
    done <<< "$MATCHES"
fi

# Pattern 2: private_key / secret_key assignment
if [ $FOUND -eq 0 ]; then
    if echo "$COMMAND" | grep -qiE '(private_key|secret_key)\s*[=:]\s*"?0?x?[0-9a-fA-F]{16,}'; then
        FOUND=1
        REASON="Private/secret key assignment detected"
    fi
fi

# Pattern 3: Common cloud/API secrets
if [ $FOUND -eq 0 ]; then
    if echo "$COMMAND" | grep -qE 'AKIA[0-9A-Z]{16}'; then
        FOUND=1
        REASON="AWS access key detected"
    fi
fi

if [ $FOUND -eq 0 ]; then
    if echo "$COMMAND" | grep -qE 'ghp_[a-zA-Z0-9]{36,}'; then
        FOUND=1
        REASON="GitHub personal access token detected"
    fi
fi

if [ $FOUND -eq 0 ]; then
    if echo "$COMMAND" | grep -qE 'sk_live_[a-zA-Z0-9]{20,}'; then
        FOUND=1
        REASON="Stripe live key detected"
    fi
fi

# Block if secret found
if [ $FOUND -eq 1 ]; then
    jq -n --arg reason "$REASON. Remove sensitive data before posting to GitHub." '{
        decision: "block",
        reason: $reason
    }'
    exit 0
fi

exit 0

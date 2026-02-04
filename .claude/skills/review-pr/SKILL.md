---
name: review-pr
description: Review a GitHub PR from the other agent. Use when PRs are open, at integration checkpoints, or before deploying contracts to mainnet. Handles directory ownership checks, schema validation, and contract security review.
---

# Review PR

Review a pull request, with awareness of the two-agent collaboration model.

## Arguments

Takes an optional PR number: `/review-pr 3`

If no number given, list open PRs and review the most recent one:
```bash
gh pr list --state open
```

## Steps

### 1. Get the PR

```bash
gh pr view <number> --json title,body,author,files,additions,deletions
gh pr diff <number>
```

### 2. Directory ownership check

This project has two agents with directory ownership:

| Directory | Owner |
|-----------|-------|
| `nojohns/`, `games/`, `arena/`, `fighters/` | **Scav** |
| `contracts/`, `web/` | **ScavieFae** |
| `docs/`, `tests/`, root files | **Shared** |

Flag if a PR touches directories outside the author's ownership. This isn't necessarily wrong (shared dirs are fine, CLAUDE.md updates happen), but note it in the review.

### 3. Review criteria

**For all PRs:**
- Does the code do what the PR description says?
- Any obvious bugs, logic errors, or missing error handling?
- Does it follow existing patterns in the codebase?
- Are there test changes? Should there be?

**For Solidity PRs (contracts/) — extra scrutiny:**
- Reentrancy: does any function send ETH/MON before updating state?
- Access control: who can call each function? Are there missing `onlyOwner` or signature checks?
- Integer overflow: is unchecked math used safely?
- Escrow safety: can funds get locked permanently? Are there escape hatches (timeout, cancel)?
- EIP-712 signature validation: are signatures checked against the correct struct hash and domain separator?
- Match the shared `MatchResult` struct (defined in `contracts/CLAUDE.md`):
  ```solidity
  struct MatchResult {
      bytes32 matchId;
      address winner;
      address loser;
      string gameId;
      uint8 winnerScore;
      uint8 loserScore;
      bytes32 replayHash;
      uint256 timestamp;
  }
  ```
  If the struct has changed, flag it — the Python signing code must match.
- Gas: anything unusually expensive? (Loops over unbounded arrays, excessive storage writes)
- Events: are key state changes emitted? The website indexes these.

**For Python PRs (nojohns/, games/, arena/):**
- Does it break existing CLI commands?
- Are new dependencies justified?
- Does signing code match the contract's EIP-712 domain and struct?

**For website PRs (web/):**
- Does it read from the correct contract addresses?
- Does it handle RPC errors gracefully?
- Lighter review — website is read-only, low risk.

### 4. Schema consistency check

If the PR touches anything related to match results, signatures, or contract interaction, verify the `MatchResult` struct is consistent across:
- `contracts/src/MatchProof.sol` (Solidity definition)
- `contracts/CLAUDE.md` (documented spec)
- Python signing code in `nojohns/` (if it exists yet)

Mismatches here cause silent failures — signatures won't verify.

### 5. Post the review

Use `gh pr review` to post:

```bash
# Approve
gh pr review <number> --approve --body "..."

# Request changes
gh pr review <number> --request-changes --body "..."

# Comment only (no approval/rejection)
gh pr review <number> --comment --body "..."
```

**Review format:**

```markdown
## Review: <PR title>

**Ownership:** [OK — touches only author's directories / NOTE — touches shared dirs / WARNING — touches other agent's directories]

**Schema:** [OK — consistent / WARNING — MatchResult struct changed / N/A — no schema-related changes]

### Findings

- [list of observations, concerns, suggestions]

### Verdict

[approve / request changes / comment]
```

### 6. Contract deploy gate

If the PR includes Solidity changes AND targets mainnet deployment:
- Do NOT approve without reading every line of the contract
- Verify forge tests pass: `cd contracts && forge test -vv`
- Check deployment script uses correct RPC and chain ID
- Flag any contract that hasn't been tested on testnet first

Testnet deploys can be approved with lighter review.

## What NOT to do

- Don't edit files in the PR. If you see something that needs changing, request changes and describe what's needed.
- Don't block PRs over style. This is a hackathon.
- Don't review your own PRs with this skill — that defeats the purpose.

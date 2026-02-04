# contracts/ — CLAUDE.md

ScavieFae owns this directory. Scav should not edit files here.

## What to Build

Two contracts, deliberately thin. The ERC-8004 registries (Identity, Reputation) are already deployed on Monad — we interact with them, we don't redeploy them.

### 1. MatchProof.sol

Records match results onchain. The core primitive everything else reads from.

```solidity
struct MatchResult {
    bytes32 matchId;
    address winner;
    address loser;
    string gameId;        // "melee", "chess", etc.
    uint8 winnerScore;
    uint8 loserScore;
    bytes32 replayHash;   // keccak256 of replay file(s)
    uint256 timestamp;
}
```

**Key functions:**
- `recordMatch(MatchResult calldata result, bytes calldata sigA, bytes calldata sigB)` — verifies both EIP-712 signatures match the result, stores proof, emits `MatchRecorded` event
- `getMatch(bytes32 matchId) → MatchResult` — read a recorded match
- `getMatchesByAgent(address agent) → bytes32[]` — match history for an agent

**EIP-712 domain:**
```solidity
EIP712Domain({
    name: "NoJohns",
    version: "1",
    chainId: <chain-id>,
    verifyingContract: <matchproof-address>
})
```

**Design notes:**
- Both signatures must be from different addresses (no self-play proofs)
- Both signatures must recover to either winner or loser (no third-party submissions... yet)
- The contract is game-agnostic. `gameId` is a string. Don't reference Melee, Slippi, or any specific game.
- Match IDs should be unique. The arena generates them (UUIDs). Contract should reject duplicate matchIds.
- Emit rich events — the website indexes these for match history.

### 2. Wager.sol

Escrow and settlement. Reads from MatchProof for trustless settlement.

**Key functions:**
- `proposeWager(address opponent, string gameId, uint256 amount)` — escrow MON, returns wagerId. If `opponent` is `address(0)`, open wager (anyone can accept).
- `acceptWager(uint256 wagerId)` — opponent escrows matching amount
- `settleWager(uint256 wagerId, bytes32 matchId)` — reads MatchProof, pays winner. Reverts if matchId doesn't have a recorded result, or if winner/loser don't match the wager participants.
- `cancelWager(uint256 wagerId)` — only before accepted. Returns escrow.
- `claimTimeout(uint256 wagerId)` — if accepted but no match result after timeout (e.g., 1 hour), both sides get refunds. Match is void.

**Design notes:**
- Wager amounts are in native MON (msg.value). No ERC-20 for now — keeps it simple.
- The timeout protects against matches that never happen. Don't make it too short (network issues) or too long (locked funds). Start with 1 hour.
- Wager contract does NOT update Elo or post reputation signals. That's a separate concern handled by the arena server calling the ReputationRegistry. Keep wagers and ranking decoupled.

### 3. Optional: ERC-8004 Integration Helpers

If time permits, a thin wrapper or script that:
- Registers an agent on the IdentityRegistry (mints NFT, sets tokenURI)
- Posts an Elo update to the ReputationRegistry after match settlement

These can be Forge scripts rather than contracts if that's simpler.

## ERC-8004 Addresses (Already Deployed)

**Monad Mainnet (chain 143):**
- IdentityRegistry: `0x8004A169FB4a3325136EB29fA0ceB6D2e539a432`
- ReputationRegistry: `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63`

**Monad Testnet (chain 10143):**
- IdentityRegistry: `0x8004A818BFB912233c491871b3d84c89A494BD9e`
- ReputationRegistry: `0x8004B663056A597Dffe9eCcC1965A193B7388713`

You'll need the ERC-8004 interfaces to call these. The contracts repo is at:
https://github.com/erc-8004/erc-8004-contracts

Install via forge:
```bash
forge install erc-8004/erc-8004-contracts
```

Or copy the interfaces manually if forge install has issues.

## Foundry Setup

```bash
# Already configured in foundry.toml:
# solc = 0.8.24
# Monad RPC endpoints read from env vars

# Set env vars:
export MONAD_TESTNET_RPC_URL=https://testnet-rpc.monad.xyz
export MONAD_MAINNET_RPC_URL=https://rpc.monad.xyz

# Build
cd contracts && forge build

# Test
cd contracts && forge test -vv

# Deploy to testnet
cd contracts && forge script script/Deploy.s.sol --rpc-url $MONAD_TESTNET_RPC_URL --broadcast
```

## Deploy Output

After deploying, create `contracts/deployments.json`:

```json
{
  "monad_testnet": {
    "chain_id": 10143,
    "match_proof": "0x...",
    "wager": "0x...",
    "deployed_at": "2026-02-XX",
    "deployer": "0x..."
  },
  "monad_mainnet": {
    "chain_id": 143,
    "match_proof": "0x...",
    "wager": "0x...",
    "deployed_at": "2026-02-XX",
    "deployer": "0x..."
  }
}
```

Scav reads this file to wire up the Python integration.

## What NOT to Do

- Don't reference Melee, Slippi, or any specific game in contract code. `gameId` is a string.
- Don't couple wagering with Elo/reputation updates. Keep them separate.
- Don't over-engineer. Two contracts, a deploy script, and tests. Ship it.
- Don't edit files outside `contracts/` — that's Scav's territory.

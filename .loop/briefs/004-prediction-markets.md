# Brief: Spectator Prediction Markets

**Branch:** scav/prediction-markets
**Model:** sonnet

## Goal

One scan, one tap, you're betting. A spectator at Fight Night scans a QR code, signs in with email via Privy, gets a funded embedded wallet, and bets on the match in front of them — all without knowing what a blockchain is.

The contract is deployed. Web components exist. Arena auto-creates pools. The gap is the 30-second onboarding flow.

Reference: `web/src/components/prediction/`, `nojohns/contract.py`, `contracts/src/PredictionPool.sol`, `tournaments/PROGRAM.md`.

## User Story

**Non-crypto attendee:**
1. Walks in. Sees bracket screen: "MARTH vs FOX — BETTING OPEN" with a QR code.
2. Scans QR. Browser opens `tournaments.nojohns.gg/bet`.
3. Page says "Sign in to bet." Taps email/Google. Privy creates an embedded wallet.
4. Page shows: two fighters, an odds bar, two big buttons — "Bet on Marth" / "Bet on Fox." No dollar signs, no token names.
5. Taps one. Tx goes through invisibly. "You're in!"
6. Match plays. Fox wins. "You won!" with a Claim button, or "Better luck next match."
7. Page auto-shows the next match. Balance shows "2 bets remaining" (not "0.1 MON").
8. One scan at the door. Then it just works all night.

**Crypto-native attendee:**
1. Same QR, same page. Taps "Connect Wallet" instead of email sign-in.
2. MetaMask/Rabby connects. Bets from their own funds.
3. Same flow from there.

## Design Decisions

- **Privy for wallet onboarding.** Embedded wallets via email/social login. No MetaMask required, no private keys in URLs. Privy handles key management.
- **Fund embedded wallets.** After Privy creates a wallet, we fund it with 0.1 MON from an operator wallet. Script to pre-fund or on-demand via a faucet endpoint.
- **Fixed bet amounts, not free-form.** Two buttons: "0.05 on [Player A]" / "0.05 on [Player B]." Sports betting UX, not DeFi. With 0.1 MON funding, each person gets ~2 bets for the night.
- **One page for the whole night.** `/bet` shows whatever match is currently live. Auto-updates between matches. No per-match QR scanning after the first one.
- **No visible crypto.** No gas estimates, no tx hashes, no "approve token" steps. Privy + gasless relaying if possible, or just eat the gas from the embedded wallet's balance.
- **Standalone product potential.** This generalizes beyond Melee — any live head-to-head event. Note for future, not for Wednesday.

## Tasks

1. Integrate Privy into the web app (`web/`). Add `@privy-io/react-auth` provider. Support email sign-in → embedded wallet creation. Also support external wallet connect (MetaMask/Rabby) for crypto-native users.

2. Build `/bet` page. Shows: current match (names + characters from arena), odds bar, two fixed-amount bet buttons. Polls arena for current tournament match + pool state. Auto-transitions between matches. Feels like a sports betting app.

3. Add claim flow. "Claim Payout" button after match resolves (winner's side). "Claim Refund" button if pool is cancelled. Both call the PredictionPool contract.

4. Add QR code to bracket viewer (`tournaments/viewer.html`). Single QR linking to `/bet`. Displayed prominently — this is the only thing spectators need to scan.

5. Create `scripts/fund_spectator_wallets.py` — operator script that reads Privy-created wallet addresses (from Privy dashboard or an API call) and batch-funds them with 0.1 MON each. Alternatively: a `/faucet` endpoint in the arena that sends 0.1 MON to any new Privy wallet on first sign-in (capped at 30 wallets).

6. Add a wallet balance display on the bet page. Shows remaining MON. When balance hits 0, show "You're out of funds" with a friendly message (not an error).

## Resolved Decisions

- **Faucet on first sign-in.** Auto-send funds when a new Privy wallet is created. Better UX than batch-funding. Cap at 50 wallets to bound exposure.
- **Hide crypto units.** The audience doesn't know what a MON is. Frame bets as "Bet on Marth" / "Bet on Fox" — fixed amount behind the scenes, no denomination visible. Show balance as a simple number or "2 bets remaining."
- **Fixed bet size.** One tap = one bet = one unit. No amount picker. Keep it dead simple.

## Open Questions

- **Privy on Monad** — does Privy support Monad mainnet (chain 143)? Need to research docs. Never implemented before.
- **Gas sponsorship** — can we use Privy's paymaster so bets are truly gasless? Or fund enough to cover gas too?
- **Privy pricing** — free tier sufficient for 30-50 users at a one-night event?

## Completion Criteria

- [ ] Privy sign-in works (email → embedded wallet)
- [ ] `/bet` page shows current match and accepts bets with one tap
- [ ] External wallet connect works for crypto-native users
- [ ] New users get funded with 0.1 MON (faucet or batch script)
- [ ] Claim payout and claim refund buttons work
- [ ] QR code on bracket viewer links to `/bet`
- [ ] Page auto-shows next match after resolution

## Verification

- `cd web && npm run build` succeeds
- Manual test: scan QR → sign in with email → wallet created → funded → place bet → bet appears in pool
- Manual test: cancel pool → "Claim Refund" appears → refund succeeds
- Page renders clean on mobile (most spectators will be on phones)

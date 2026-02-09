// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {MatchProof} from "../src/MatchProof.sol";
import {PredictionPool} from "../src/PredictionPool.sol";

contract PredictionPoolTest is Test {
    MatchProof public proof;
    PredictionPool public pool;

    uint256 internal playerAKey = 0xA1;
    uint256 internal playerBKey = 0xB2;
    address internal playerA;
    address internal playerB;
    address internal arena;

    address internal bettor1;
    address internal bettor2;
    address internal bettor3;

    uint256 constant FEE_BPS = 0; // No fee by default

    function setUp() public {
        proof = new MatchProof();
        arena = makeAddr("arena");
        playerA = vm.addr(playerAKey);
        playerB = vm.addr(playerBKey);

        pool = new PredictionPool(address(proof), arena, FEE_BPS);

        bettor1 = makeAddr("bettor1");
        bettor2 = makeAddr("bettor2");
        bettor3 = makeAddr("bettor3");

        vm.deal(bettor1, 100 ether);
        vm.deal(bettor2, 100 ether);
        vm.deal(bettor3, 100 ether);
    }

    // -- Helpers --

    function _makeResult(bytes32 matchId, address winner, address loser)
        internal view returns (MatchProof.MatchResult memory)
    {
        return MatchProof.MatchResult({
            matchId: matchId,
            winner: winner,
            loser: loser,
            gameId: "melee",
            winnerScore: 3,
            loserScore: 1,
            replayHash: keccak256("replay"),
            timestamp: block.timestamp
        });
    }

    function _signResult(MatchProof.MatchResult memory result, uint256 key)
        internal view returns (bytes memory)
    {
        bytes32 digest = proof.getDigest(result);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(key, digest);
        return abi.encodePacked(r, s, v);
    }

    function _recordMatch(bytes32 matchId) internal {
        MatchProof.MatchResult memory result = _makeResult(matchId, playerA, playerB);
        bytes memory sigA = _signResult(result, playerAKey);
        bytes memory sigB = _signResult(result, playerBKey);
        proof.recordMatch(result, sigA, sigB);
    }

    function _recordMatchBWins(bytes32 matchId) internal {
        MatchProof.MatchResult memory result = _makeResult(matchId, playerB, playerA);
        bytes memory sigA = _signResult(result, playerAKey);
        bytes memory sigB = _signResult(result, playerBKey);
        proof.recordMatch(result, sigA, sigB);
    }

    function _createPool(bytes32 matchId) internal returns (uint256) {
        vm.prank(arena);
        return pool.createPool(matchId, playerA, playerB);
    }

    // ==================== Create Pool ====================

    function test_createPool() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(p.matchId, bytes32("match-1"));
        assertEq(p.playerA, playerA);
        assertEq(p.playerB, playerB);
        assertEq(p.totalA, 0);
        assertEq(p.totalB, 0);
        assertEq(uint8(p.status), uint8(PredictionPool.PoolStatus.Open));
        assertEq(p.winner, address(0));
        assertEq(pool.poolCount(), 1);
    }

    function test_createPool_emitsEvent() public {
        vm.prank(arena);
        vm.expectEmit(true, true, true, true);
        emit PredictionPool.PoolCreated(0, bytes32("match-1"), playerA, playerB);
        pool.createPool(bytes32("match-1"), playerA, playerB);
    }

    function test_revert_createPool_notArena() public {
        vm.expectRevert(PredictionPool.OnlyArena.selector);
        pool.createPool(bytes32("match-1"), playerA, playerB);
    }

    function test_revert_createPool_samePlayer() public {
        vm.prank(arena);
        vm.expectRevert("Same player");
        pool.createPool(bytes32("match-1"), playerA, playerA);
    }

    function test_revert_createPool_zeroAddress() public {
        vm.prank(arena);
        vm.expectRevert("Zero address");
        pool.createPool(bytes32("match-1"), address(0), playerB);
    }

    // ==================== Bet ====================

    function test_bet_onA() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(p.totalA, 1 ether);
        assertEq(p.totalB, 0);

        (uint256 onA, uint256 onB) = pool.getUserBets(poolId, bettor1);
        assertEq(onA, 1 ether);
        assertEq(onB, 0);
    }

    function test_bet_onB() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        vm.prank(bettor1);
        pool.bet{value: 2 ether}(poolId, false);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(p.totalA, 0);
        assertEq(p.totalB, 2 ether);
    }

    function test_bet_multipleBettors() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);

        vm.prank(bettor2);
        pool.bet{value: 3 ether}(poolId, false);

        vm.prank(bettor3);
        pool.bet{value: 2 ether}(poolId, true);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(p.totalA, 3 ether);
        assertEq(p.totalB, 3 ether);
    }

    function test_bet_sameUserMultipleTimes() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        vm.startPrank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        pool.bet{value: 0.5 ether}(poolId, true);
        vm.stopPrank();

        (uint256 onA,) = pool.getUserBets(poolId, bettor1);
        assertEq(onA, 1.5 ether);
    }

    function test_bet_emitsEvent() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        vm.prank(bettor1);
        vm.expectEmit(true, true, false, true);
        emit PredictionPool.BetPlaced(poolId, bettor1, true, 1 ether);
        pool.bet{value: 1 ether}(poolId, true);
    }

    function test_revert_bet_zeroBet() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.ZeroBet.selector);
        pool.bet{value: 0}(poolId, true);
    }

    function test_revert_bet_poolNotOpen() public {
        uint256 poolId = _createPool(bytes32("match-1"));

        // Cancel the pool
        vm.prank(arena);
        pool.cancelPool(poolId);

        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.PoolNotOpen.selector);
        pool.bet{value: 1 ether}(poolId, true);
    }

    function test_revert_bet_poolNotFound() public {
        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.PoolNotFound.selector);
        pool.bet{value: 1 ether}(999, true);
    }

    // ==================== Resolve ====================

    function test_resolve_playerAWins() public {
        bytes32 matchId = bytes32("match-resolve-a");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 1 ether}(poolId, false);

        _recordMatch(matchId); // playerA wins

        pool.resolve(poolId);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(uint8(p.status), uint8(PredictionPool.PoolStatus.Resolved));
        assertEq(p.winner, playerA);
    }

    function test_resolve_playerBWins() public {
        bytes32 matchId = bytes32("match-resolve-b");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 1 ether}(poolId, false);

        _recordMatchBWins(matchId);

        pool.resolve(poolId);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(p.winner, playerB);
    }

    function test_resolve_emitsEvent() public {
        bytes32 matchId = bytes32("match-emit");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 2 ether}(poolId, false);

        _recordMatch(matchId);

        vm.expectEmit(true, true, false, true);
        emit PredictionPool.PoolResolved(poolId, playerA, 3 ether, 0);
        pool.resolve(poolId);
    }

    function test_revert_resolve_matchNotRecorded() public {
        uint256 poolId = _createPool(bytes32("no-match"));

        vm.expectRevert(PredictionPool.MatchNotRecorded.selector);
        pool.resolve(poolId);
    }

    function test_revert_resolve_alreadyResolved() public {
        bytes32 matchId = bytes32("match-double");
        uint256 poolId = _createPool(matchId);
        _recordMatch(matchId);
        pool.resolve(poolId);

        vm.expectRevert(PredictionPool.PoolNotOpen.selector);
        pool.resolve(poolId);
    }

    // ==================== Claim ====================

    function test_claim_proportionalPayout() public {
        bytes32 matchId = bytes32("match-claim");
        uint256 poolId = _createPool(matchId);

        // bettor1: 1 ETH on A, bettor2: 3 ETH on B, bettor3: 3 ETH on A
        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 3 ether}(poolId, false);
        vm.prank(bettor3);
        pool.bet{value: 3 ether}(poolId, true);

        _recordMatch(matchId); // playerA wins
        pool.resolve(poolId);

        // Total pool = 7 ETH, A side = 4 ETH, B side = 3 ETH
        // bettor1 gets 1/4 * 7 = 1.75 ETH
        // bettor3 gets 3/4 * 7 = 5.25 ETH
        uint256 bal1Before = bettor1.balance;
        uint256 bal3Before = bettor3.balance;

        vm.prank(bettor1);
        pool.claim(poolId);
        vm.prank(bettor3);
        pool.claim(poolId);

        assertEq(bettor1.balance, bal1Before + 1.75 ether);
        assertEq(bettor3.balance, bal3Before + 5.25 ether);
    }

    function test_claim_onlyWinningSideBets() public {
        bytes32 matchId = bytes32("match-claim-2");
        uint256 poolId = _createPool(matchId);

        // bettor1 bets on both sides
        vm.startPrank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        pool.bet{value: 2 ether}(poolId, false);
        vm.stopPrank();

        _recordMatch(matchId); // playerA wins
        pool.resolve(poolId);

        // bettor1 bet 1 ETH on winning side (A), total A = 1 ETH, total pool = 3 ETH
        // Payout = (3 * 1) / 1 = 3 ETH (gets everything since they're the only A bettor)
        uint256 balBefore = bettor1.balance;
        vm.prank(bettor1);
        pool.claim(poolId);
        assertEq(bettor1.balance, balBefore + 3 ether);
    }

    function test_claim_emitsEvent() public {
        bytes32 matchId = bytes32("match-claim-evt");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 1 ether}(poolId, false);

        _recordMatch(matchId);
        pool.resolve(poolId);

        vm.prank(bettor1);
        vm.expectEmit(true, true, false, true);
        emit PredictionPool.PayoutClaimed(poolId, bettor1, 2 ether);
        pool.claim(poolId);
    }

    function test_revert_claim_notResolved() public {
        uint256 poolId = _createPool(bytes32("match-nr"));

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);

        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.PoolNotResolved.selector);
        pool.claim(poolId);
    }

    function test_revert_claim_alreadyClaimed() public {
        bytes32 matchId = bytes32("match-dbl-claim");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 1 ether}(poolId, false);

        _recordMatch(matchId);
        pool.resolve(poolId);

        vm.startPrank(bettor1);
        pool.claim(poolId);
        vm.expectRevert(PredictionPool.AlreadyClaimed.selector);
        pool.claim(poolId);
        vm.stopPrank();
    }

    function test_revert_claim_noPayout_losingBettor() public {
        bytes32 matchId = bytes32("match-loser");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 1 ether}(poolId, false);

        _recordMatch(matchId); // A wins
        pool.resolve(poolId);

        // bettor2 bet on B (the loser)
        vm.prank(bettor2);
        vm.expectRevert(PredictionPool.NoPayout.selector);
        pool.claim(poolId);
    }

    // ==================== Cancel ====================

    function test_cancelPool() public {
        uint256 poolId = _createPool(bytes32("match-cancel"));

        vm.prank(arena);
        pool.cancelPool(poolId);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(uint8(p.status), uint8(PredictionPool.PoolStatus.Cancelled));
    }

    function test_revert_cancelPool_notArena() public {
        uint256 poolId = _createPool(bytes32("match-cancel-2"));

        vm.expectRevert(PredictionPool.OnlyArena.selector);
        pool.cancelPool(poolId);
    }

    function test_revert_cancelPool_notOpen() public {
        bytes32 matchId = bytes32("match-cancel-3");
        uint256 poolId = _createPool(matchId);
        _recordMatch(matchId);
        pool.resolve(poolId);

        vm.prank(arena);
        vm.expectRevert(PredictionPool.PoolNotOpen.selector);
        pool.cancelPool(poolId);
    }

    // ==================== Refund ====================

    function test_claimRefund() public {
        uint256 poolId = _createPool(bytes32("match-refund"));

        vm.prank(bettor1);
        pool.bet{value: 2 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 3 ether}(poolId, false);

        vm.prank(arena);
        pool.cancelPool(poolId);

        uint256 bal1Before = bettor1.balance;
        uint256 bal2Before = bettor2.balance;

        vm.prank(bettor1);
        pool.claimRefund(poolId);
        vm.prank(bettor2);
        pool.claimRefund(poolId);

        assertEq(bettor1.balance, bal1Before + 2 ether);
        assertEq(bettor2.balance, bal2Before + 3 ether);
    }

    function test_claimRefund_bothSides() public {
        // User who bet on both sides gets full refund
        uint256 poolId = _createPool(bytes32("match-refund-both"));

        vm.startPrank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        pool.bet{value: 2 ether}(poolId, false);
        vm.stopPrank();

        vm.prank(arena);
        pool.cancelPool(poolId);

        uint256 balBefore = bettor1.balance;
        vm.prank(bettor1);
        pool.claimRefund(poolId);
        assertEq(bettor1.balance, balBefore + 3 ether);
    }

    function test_revert_claimRefund_notCancelled() public {
        uint256 poolId = _createPool(bytes32("match-refund-2"));

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);

        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.PoolNotClaimable.selector);
        pool.claimRefund(poolId);
    }

    function test_revert_claimRefund_noBet() public {
        uint256 poolId = _createPool(bytes32("match-refund-3"));

        vm.prank(arena);
        pool.cancelPool(poolId);

        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.NoBetToRefund.selector);
        pool.claimRefund(poolId);
    }

    function test_revert_claimRefund_alreadyClaimed() public {
        uint256 poolId = _createPool(bytes32("match-refund-4"));

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);

        vm.prank(arena);
        pool.cancelPool(poolId);

        vm.startPrank(bettor1);
        pool.claimRefund(poolId);
        vm.expectRevert(PredictionPool.AlreadyClaimed.selector);
        pool.claimRefund(poolId);
        vm.stopPrank();
    }

    // ==================== Stale Pool Timeout ====================

    function test_cancelStalePool() public {
        uint256 poolId = _createPool(bytes32("match-stale"));

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);

        vm.warp(block.timestamp + 2 hours + 1);

        pool.cancelStalePool(poolId);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(uint8(p.status), uint8(PredictionPool.PoolStatus.Cancelled));
    }

    function test_revert_cancelStalePool_tooEarly() public {
        uint256 poolId = _createPool(bytes32("match-stale-2"));

        vm.expectRevert(PredictionPool.TimeoutNotReached.selector);
        pool.cancelStalePool(poolId);
    }

    // ==================== Fee ====================

    function test_resolve_withFee() public {
        // Deploy a pool with 5% fee
        PredictionPool feePool = new PredictionPool(address(proof), arena, 500);

        bytes32 matchId = bytes32("match-fee");
        vm.prank(arena);
        uint256 poolId = feePool.createPool(matchId, playerA, playerB);

        vm.prank(bettor1);
        feePool.bet{value: 4 ether}(poolId, true);
        vm.prank(bettor2);
        feePool.bet{value: 6 ether}(poolId, false);

        _recordMatch(matchId); // playerA wins
        feePool.resolve(poolId);

        // Total pool = 10 ETH, fee = 5% = 0.5 ETH, distributable = 9.5 ETH
        // bettor1 bet 4 ETH on A (only A bettor), gets all 9.5 ETH
        assertEq(feePool.accumulatedFees(), 0.5 ether);

        uint256 balBefore = bettor1.balance;
        vm.prank(bettor1);
        feePool.claim(poolId);
        assertEq(bettor1.balance, balBefore + 9.5 ether);
    }

    function test_resolve_zeroFee() public {
        // Default pool has 0% fee
        bytes32 matchId = bytes32("match-no-fee");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 5 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 5 ether}(poolId, false);

        _recordMatch(matchId);
        pool.resolve(poolId);

        assertEq(pool.accumulatedFees(), 0);

        uint256 balBefore = bettor1.balance;
        vm.prank(bettor1);
        pool.claim(poolId);
        assertEq(bettor1.balance, balBefore + 10 ether);
    }

    function test_withdrawFees() public {
        PredictionPool feePool = new PredictionPool(address(proof), arena, 500);

        bytes32 matchId = bytes32("match-withdraw");
        vm.prank(arena);
        uint256 poolId = feePool.createPool(matchId, playerA, playerB);

        vm.prank(bettor1);
        feePool.bet{value: 5 ether}(poolId, true);
        vm.prank(bettor2);
        feePool.bet{value: 5 ether}(poolId, false);

        _recordMatch(matchId);
        feePool.resolve(poolId);

        address treasury = makeAddr("treasury");
        uint256 fees = feePool.accumulatedFees();
        assertGt(fees, 0);

        vm.prank(arena);
        feePool.withdrawFees(treasury);

        assertEq(treasury.balance, fees);
        assertEq(feePool.accumulatedFees(), 0);
    }

    function test_revert_withdrawFees_notArena() public {
        vm.expectRevert(PredictionPool.OnlyArena.selector);
        pool.withdrawFees(bettor1);
    }

    function test_revert_constructor_feeTooHigh() public {
        vm.expectRevert("Fee too high");
        new PredictionPool(address(proof), arena, 1001);
    }

    // ==================== Edge Cases ====================

    function test_oneSidedPool_winnerSideOnly() public {
        // Only bets on A, A wins → everyone gets their money back
        bytes32 matchId = bytes32("match-one-side");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 2 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 3 ether}(poolId, true);

        _recordMatch(matchId); // playerA wins
        pool.resolve(poolId);

        // Total = 5 ETH, all on A. bettor1 gets 2/5*5 = 2, bettor2 gets 3/5*5 = 3
        uint256 bal1Before = bettor1.balance;
        uint256 bal2Before = bettor2.balance;

        vm.prank(bettor1);
        pool.claim(poolId);
        vm.prank(bettor2);
        pool.claim(poolId);

        assertEq(bettor1.balance, bal1Before + 2 ether);
        assertEq(bettor2.balance, bal2Before + 3 ether);
    }

    function test_oneSidedPool_loserSideOnly() public {
        // Only bets on B, A wins → no one can claim (money stuck, but also no winners)
        bytes32 matchId = bytes32("match-one-side-2");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, false); // bet on B

        _recordMatch(matchId); // playerA wins
        pool.resolve(poolId);

        // bettor1 bet on B (loser), can't claim
        vm.prank(bettor1);
        vm.expectRevert(PredictionPool.NoPayout.selector);
        pool.claim(poolId);
    }

    function test_emptyPool_resolve() public {
        // No bets, resolve is still valid
        bytes32 matchId = bytes32("match-empty");
        uint256 poolId = _createPool(matchId);

        _recordMatch(matchId);
        pool.resolve(poolId);

        PredictionPool.Pool memory p = pool.getPool(poolId);
        assertEq(uint8(p.status), uint8(PredictionPool.PoolStatus.Resolved));
    }

    function test_getClaimable_resolved() public {
        bytes32 matchId = bytes32("match-claimable");
        uint256 poolId = _createPool(matchId);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId, true);
        vm.prank(bettor2);
        pool.bet{value: 3 ether}(poolId, false);

        _recordMatch(matchId); // A wins
        pool.resolve(poolId);

        // bettor1 gets all 4 ETH (only A bettor)
        assertEq(pool.getClaimable(poolId, bettor1), 4 ether);
        // bettor2 bet on losing side
        assertEq(pool.getClaimable(poolId, bettor2), 0);

        // After claiming, claimable drops to 0
        vm.prank(bettor1);
        pool.claim(poolId);
        assertEq(pool.getClaimable(poolId, bettor1), 0);
    }

    function test_getClaimable_cancelled() public {
        uint256 poolId = _createPool(bytes32("match-claimable-c"));

        vm.prank(bettor1);
        pool.bet{value: 2 ether}(poolId, true);

        vm.prank(arena);
        pool.cancelPool(poolId);

        assertEq(pool.getClaimable(poolId, bettor1), 2 ether);

        vm.prank(bettor1);
        pool.claimRefund(poolId);
        assertEq(pool.getClaimable(poolId, bettor1), 0);
    }

    function test_multiplePoolsIndependent() public {
        bytes32 matchId1 = bytes32("match-multi-1");
        bytes32 matchId2 = bytes32("match-multi-2");
        uint256 poolId1 = _createPool(matchId1);
        uint256 poolId2 = _createPool(matchId2);

        vm.prank(bettor1);
        pool.bet{value: 1 ether}(poolId1, true);
        vm.prank(bettor1);
        pool.bet{value: 2 ether}(poolId2, false);

        (uint256 onA1, uint256 onB1) = pool.getUserBets(poolId1, bettor1);
        (uint256 onA2, uint256 onB2) = pool.getUserBets(poolId2, bettor1);

        assertEq(onA1, 1 ether);
        assertEq(onB1, 0);
        assertEq(onA2, 0);
        assertEq(onB2, 2 ether);
    }
}

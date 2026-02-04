// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {MatchProof} from "../src/MatchProof.sol";
import {Wager} from "../src/Wager.sol";

contract WagerTest is Test {
    MatchProof public proof;
    Wager public wager;

    uint256 internal winnerKey = 0xA1;
    uint256 internal loserKey = 0xB2;
    address internal winner;
    address internal loser;

    uint256 constant BET = 1 ether;

    function setUp() public {
        proof = new MatchProof();
        wager = new Wager(address(proof));
        winner = vm.addr(winnerKey);
        loser = vm.addr(loserKey);
        vm.deal(winner, 100 ether);
        vm.deal(loser, 100 ether);
    }

    // -- Helpers --

    function _makeResult(bytes32 matchId) internal view returns (MatchProof.MatchResult memory) {
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

    function _signResult(MatchProof.MatchResult memory result, uint256 key) internal view returns (bytes memory) {
        bytes32 digest = proof.getDigest(result);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(key, digest);
        return abi.encodePacked(r, s, v);
    }

    function _recordMatch(bytes32 matchId) internal {
        MatchProof.MatchResult memory result = _makeResult(matchId);
        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, loserKey);
        proof.recordMatch(result, sigA, sigB);
    }

    function _proposeWager() internal returns (uint256) {
        vm.prank(winner);
        return wager.proposeWager{value: BET}(loser, "melee");
    }

    function _proposeOpenWager() internal returns (uint256) {
        vm.prank(winner);
        return wager.proposeWager{value: BET}(address(0), "melee");
    }

    function _acceptWager(uint256 wagerId) internal {
        vm.prank(loser);
        wager.acceptWager{value: BET}(wagerId);
    }

    // -- Propose --

    function test_proposeWager() public {
        uint256 wagerId = _proposeWager();

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(w.proposer, winner);
        assertEq(w.opponent, loser);
        assertEq(w.amount, BET);
        assertEq(uint8(w.status), uint8(Wager.WagerStatus.Open));
        assertEq(address(wager).balance, BET);
    }

    function test_proposeOpenWager() public {
        uint256 wagerId = _proposeOpenWager();

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(w.opponent, address(0));
    }

    function test_proposeWager_emitsEvent() public {
        vm.prank(winner);
        vm.expectEmit(true, true, true, true);
        emit Wager.WagerProposed(0, winner, loser, "melee", BET);
        wager.proposeWager{value: BET}(loser, "melee");
    }

    // -- Accept --

    function test_acceptWager() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(uint8(w.status), uint8(Wager.WagerStatus.Accepted));
        assertEq(w.opponent, loser);
        assertEq(address(wager).balance, BET * 2);
    }

    function test_acceptOpenWager() public {
        uint256 wagerId = _proposeOpenWager();
        _acceptWager(wagerId);

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(w.opponent, loser);
        assertEq(uint8(w.status), uint8(Wager.WagerStatus.Accepted));
    }

    function test_revert_acceptOwnWager() public {
        uint256 wagerId = _proposeWager();

        vm.prank(winner);
        vm.expectRevert(Wager.CannotAcceptOwnWager.selector);
        wager.acceptWager{value: BET}(wagerId);
    }

    function test_revert_acceptWrongAmount() public {
        uint256 wagerId = _proposeWager();

        vm.prank(loser);
        vm.expectRevert(abi.encodeWithSelector(Wager.WrongAmount.selector, BET, BET / 2));
        wager.acceptWager{value: BET / 2}(wagerId);
    }

    function test_revert_acceptWrongOpponent() public {
        uint256 wagerId = _proposeWager(); // targeted at loser

        address rando = makeAddr("rando");
        vm.deal(rando, 10 ether);
        vm.prank(rando);
        vm.expectRevert(Wager.NotOpenToYou.selector);
        wager.acceptWager{value: BET}(wagerId);
    }

    function test_revert_acceptAlreadyAccepted() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        vm.prank(loser);
        vm.expectRevert(abi.encodeWithSelector(
            Wager.WrongStatus.selector, Wager.WagerStatus.Open, Wager.WagerStatus.Accepted
        ));
        wager.acceptWager{value: BET}(wagerId);
    }

    // -- Settle --

    function test_settleWager() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        bytes32 matchId = bytes32("settle-match");
        _recordMatch(matchId);

        uint256 winnerBalBefore = winner.balance;

        wager.settleWager(wagerId, matchId);

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(uint8(w.status), uint8(Wager.WagerStatus.Settled));
        assertEq(w.matchId, matchId);
        assertEq(winner.balance, winnerBalBefore + BET * 2);
        assertEq(address(wager).balance, 0);
    }

    function test_revert_settleNoMatchRecorded() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        vm.expectRevert(Wager.MatchNotRecorded.selector);
        wager.settleWager(wagerId, bytes32("fake-match"));
    }

    function test_revert_settleParticipantMismatch() public {
        // Create wager between winner and loser
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        // Record a match between different addresses
        address otherWinner = makeAddr("other-winner");
        address otherLoser = makeAddr("other-loser");
        bytes32 matchId = bytes32("wrong-participants");

        MatchProof.MatchResult memory result = MatchProof.MatchResult({
            matchId: matchId,
            winner: otherWinner,
            loser: otherLoser,
            gameId: "melee",
            winnerScore: 3,
            loserScore: 0,
            replayHash: keccak256("replay"),
            timestamp: block.timestamp
        });

        // Need to sign with the actual participants' keys
        uint256 otherWinnerKey = 0xD4;
        uint256 otherLoserKey = 0xE5;
        // Recreate with correct addresses from these keys
        result.winner = vm.addr(otherWinnerKey);
        result.loser = vm.addr(otherLoserKey);

        bytes memory sigA = _signResult(result, otherWinnerKey);
        bytes memory sigB = _signResult(result, otherLoserKey);
        proof.recordMatch(result, sigA, sigB);

        vm.expectRevert(Wager.ParticipantMismatch.selector);
        wager.settleWager(wagerId, matchId);
    }

    function test_revert_settleNotAccepted() public {
        uint256 wagerId = _proposeWager();
        bytes32 matchId = bytes32("premature");
        _recordMatch(matchId);

        vm.expectRevert(abi.encodeWithSelector(
            Wager.WrongStatus.selector, Wager.WagerStatus.Accepted, Wager.WagerStatus.Open
        ));
        wager.settleWager(wagerId, matchId);
    }

    // -- Cancel --

    function test_cancelWager() public {
        uint256 wagerId = _proposeWager();
        uint256 balBefore = winner.balance;

        vm.prank(winner);
        wager.cancelWager(wagerId);

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(uint8(w.status), uint8(Wager.WagerStatus.Cancelled));
        assertEq(winner.balance, balBefore + BET);
    }

    function test_revert_cancelNotProposer() public {
        uint256 wagerId = _proposeWager();

        vm.prank(loser);
        vm.expectRevert(Wager.NotAuthorized.selector);
        wager.cancelWager(wagerId);
    }

    function test_revert_cancelAfterAccepted() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        vm.prank(winner);
        vm.expectRevert(abi.encodeWithSelector(
            Wager.WrongStatus.selector, Wager.WagerStatus.Open, Wager.WagerStatus.Accepted
        ));
        wager.cancelWager(wagerId);
    }

    // -- Timeout --

    function test_claimTimeout() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        uint256 winnerBal = winner.balance;
        uint256 loserBal = loser.balance;

        vm.warp(block.timestamp + 1 hours + 1);
        wager.claimTimeout(wagerId);

        Wager.WagerInfo memory w = wager.getWager(wagerId);
        assertEq(uint8(w.status), uint8(Wager.WagerStatus.Voided));
        assertEq(winner.balance, winnerBal + BET);
        assertEq(loser.balance, loserBal + BET);
    }

    function test_revert_timeoutTooEarly() public {
        uint256 wagerId = _proposeWager();
        _acceptWager(wagerId);

        vm.expectRevert(Wager.TimeoutNotReached.selector);
        wager.claimTimeout(wagerId);
    }

    function test_revert_timeoutNotAccepted() public {
        uint256 wagerId = _proposeWager();

        vm.warp(block.timestamp + 2 hours);
        vm.expectRevert(abi.encodeWithSelector(
            Wager.WrongStatus.selector, Wager.WagerStatus.Accepted, Wager.WagerStatus.Open
        ));
        wager.claimTimeout(wagerId);
    }

    // -- View helpers --

    function test_getWagersByAgent() public {
        _proposeWager();
        _proposeWager();

        uint256[] memory winnerWagers = wager.getWagersByAgent(winner);
        assertEq(winnerWagers.length, 2);
    }

    function test_getWager_nonexistent() public view {
        Wager.WagerInfo memory w = wager.getWager(999);
        assertEq(w.proposer, address(0));
        assertEq(w.amount, 0);
    }
}

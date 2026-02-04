// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {MatchProof} from "../src/MatchProof.sol";

contract MatchProofTest is Test {
    MatchProof public proof;

    uint256 internal winnerKey = 0xA1;
    uint256 internal loserKey = 0xB2;
    address internal winner;
    address internal loser;

    function setUp() public {
        proof = new MatchProof();
        winner = vm.addr(winnerKey);
        loser = vm.addr(loserKey);
    }

    function _makeResult(bytes32 matchId) internal view returns (MatchProof.MatchResult memory) {
        return MatchProof.MatchResult({
            matchId: matchId,
            winner: winner,
            loser: loser,
            gameId: "melee",
            winnerScore: 3,
            loserScore: 1,
            replayHash: keccak256("replay-data"),
            timestamp: block.timestamp
        });
    }

    function _signResult(MatchProof.MatchResult memory result, uint256 key) internal view returns (bytes memory) {
        bytes32 digest = proof.getDigest(result);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(key, digest);
        return abi.encodePacked(r, s, v);
    }

    function _recordMatch(bytes32 matchId) internal returns (MatchProof.MatchResult memory) {
        MatchProof.MatchResult memory result = _makeResult(matchId);
        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, loserKey);
        proof.recordMatch(result, sigA, sigB);
        return result;
    }

    function test_recordMatch() public {
        bytes32 matchId = bytes32("match-1");
        MatchProof.MatchResult memory result = _recordMatch(matchId);

        assertTrue(proof.recorded(matchId));
        MatchProof.MatchResult memory stored = proof.getMatch(matchId);
        assertEq(stored.winner, winner);
        assertEq(stored.loser, loser);
        assertEq(stored.winnerScore, result.winnerScore);
        assertEq(stored.loserScore, result.loserScore);
        assertEq(stored.replayHash, result.replayHash);
    }

    function test_recordMatch_emitsEvent() public {
        bytes32 matchId = bytes32("match-event");
        MatchProof.MatchResult memory result = _makeResult(matchId);
        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, loserKey);

        vm.expectEmit(true, true, true, true);
        emit MatchProof.MatchRecorded(
            matchId, winner, loser, "melee", 3, 1,
            result.replayHash, result.timestamp
        );
        proof.recordMatch(result, sigA, sigB);
    }

    function test_recordMatch_signaturesReversed() public {
        bytes32 matchId = bytes32("match-reversed");
        MatchProof.MatchResult memory result = _makeResult(matchId);
        // Swap signature order — loser signs first, winner second
        bytes memory sigA = _signResult(result, loserKey);
        bytes memory sigB = _signResult(result, winnerKey);

        proof.recordMatch(result, sigA, sigB);
        assertTrue(proof.recorded(matchId));
    }

    function test_revert_duplicateMatchId() public {
        bytes32 matchId = bytes32("match-dup");
        _recordMatch(matchId);

        MatchProof.MatchResult memory result = _makeResult(matchId);
        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, loserKey);

        vm.expectRevert(abi.encodeWithSelector(MatchProof.MatchAlreadyRecorded.selector, matchId));
        proof.recordMatch(result, sigA, sigB);
    }

    function test_revert_sameWinnerAndLoser() public {
        MatchProof.MatchResult memory result = _makeResult(bytes32("match-self"));
        result.loser = winner; // same as winner

        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, winnerKey);

        vm.expectRevert(MatchProof.SameAddress.selector);
        proof.recordMatch(result, sigA, sigB);
    }

    function test_revert_invalidSignature() public {
        uint256 randomKey = 0xC3;
        MatchProof.MatchResult memory result = _makeResult(bytes32("match-badsig"));

        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, randomKey); // not a participant

        vm.expectRevert(abi.encodeWithSelector(MatchProof.SignerNotParticipant.selector, vm.addr(randomKey)));
        proof.recordMatch(result, sigA, sigB);
    }

    function test_revert_bothSignaturesSamePerson() public {
        MatchProof.MatchResult memory result = _makeResult(bytes32("match-same-signer"));

        bytes memory sigA = _signResult(result, winnerKey);
        bytes memory sigB = _signResult(result, winnerKey);

        vm.expectRevert(MatchProof.SameAddress.selector);
        proof.recordMatch(result, sigA, sigB);
    }

    function test_getMatchesByAgent() public {
        _recordMatch(bytes32("m1"));
        _recordMatch(bytes32("m2"));

        bytes32[] memory winnerMatches = proof.getMatchesByAgent(winner);
        assertEq(winnerMatches.length, 2);

        bytes32[] memory loserMatches = proof.getMatchesByAgent(loser);
        assertEq(loserMatches.length, 2);
    }

    function test_getMatch_unrecorded() public view {
        MatchProof.MatchResult memory result = proof.getMatch(bytes32("nonexistent"));
        assertEq(result.winner, address(0));
    }
}

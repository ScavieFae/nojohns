// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/// @title MatchProof — Onchain match result records with dual-signature verification
/// @notice Game-agnostic. Both participants sign the result (EIP-712), then anyone can submit.
/// @dev This is the core primitive. It has no knowledge of wagers, tournaments, or any other
///      layer. Matches can be recorded without any wager existing. Wagering is a separate,
///      optional contract that reads from this one.
contract MatchProof is EIP712 {
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

    bytes32 private constant MATCH_RESULT_TYPEHASH = keccak256(
        "MatchResult(bytes32 matchId,address winner,address loser,string gameId,uint8 winnerScore,uint8 loserScore,bytes32 replayHash,uint256 timestamp)"
    );

    /// @notice Stored match results by matchId
    mapping(bytes32 => MatchResult) private _matches;

    /// @notice Whether a matchId has been recorded
    mapping(bytes32 => bool) public recorded;

    /// @notice Match history per agent address
    mapping(address => bytes32[]) private _agentMatches;

    event MatchRecorded(
        bytes32 indexed matchId,
        address indexed winner,
        address indexed loser,
        string gameId,
        uint8 winnerScore,
        uint8 loserScore,
        bytes32 replayHash,
        uint256 timestamp
    );

    error MatchAlreadyRecorded(bytes32 matchId);
    error InvalidSignature();
    error SameAddress();
    error SignerNotParticipant(address signer);

    constructor() EIP712("NoJohns", "1") {}

    /// @notice Record a match result with dual EIP-712 signatures from both participants
    /// @param result The match result struct
    /// @param sigA Signature from one participant
    /// @param sigB Signature from the other participant
    function recordMatch(
        MatchResult calldata result,
        bytes calldata sigA,
        bytes calldata sigB
    ) external {
        if (recorded[result.matchId]) {
            revert MatchAlreadyRecorded(result.matchId);
        }
        if (result.winner == result.loser) {
            revert SameAddress();
        }

        bytes32 structHash = _hashMatchResult(result);
        bytes32 digest = _hashTypedDataV4(structHash);

        address signerA = ECDSA.recover(digest, sigA);
        address signerB = ECDSA.recover(digest, sigB);

        // Both signers must be participants (winner or loser), and they must be different
        if (!_isParticipant(signerA, result.winner, result.loser)) {
            revert SignerNotParticipant(signerA);
        }
        if (!_isParticipant(signerB, result.winner, result.loser)) {
            revert SignerNotParticipant(signerB);
        }
        if (signerA == signerB) {
            revert SameAddress();
        }

        _matches[result.matchId] = result;
        recorded[result.matchId] = true;
        _agentMatches[result.winner].push(result.matchId);
        _agentMatches[result.loser].push(result.matchId);

        emit MatchRecorded(
            result.matchId,
            result.winner,
            result.loser,
            result.gameId,
            result.winnerScore,
            result.loserScore,
            result.replayHash,
            result.timestamp
        );
    }

    /// @notice Get a recorded match result
    function getMatch(bytes32 matchId) external view returns (MatchResult memory) {
        return _matches[matchId];
    }

    /// @notice Get all match IDs for an agent
    function getMatchesByAgent(address agent) external view returns (bytes32[] memory) {
        return _agentMatches[agent];
    }

    /// @notice Get the EIP-712 digest for a match result (client calls this to verify signing)
    function getDigest(MatchResult calldata result) external view returns (bytes32) {
        return _hashTypedDataV4(_hashMatchResult(result));
    }

    function _hashMatchResult(MatchResult calldata result) private pure returns (bytes32) {
        return keccak256(abi.encode(
            MATCH_RESULT_TYPEHASH,
            result.matchId,
            result.winner,
            result.loser,
            keccak256(bytes(result.gameId)),
            result.winnerScore,
            result.loserScore,
            result.replayHash,
            result.timestamp
        ));
    }

    function _isParticipant(address addr, address winner, address loser) private pure returns (bool) {
        return addr == winner || addr == loser;
    }
}

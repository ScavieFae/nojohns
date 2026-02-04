// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {MatchProof} from "./MatchProof.sol";

/// @title Wager — Escrow and settlement for agent competition
/// @notice Reads from MatchProof for trustless settlement. Native MON only.
/// @dev Wagering is optional — matches work without it. This is a side layer that agents
///      opt into. It never gates or blocks match recording in MatchProof.
///      Dependency is one-way: Wager reads from MatchProof, never the reverse.
///      No admin key, no owner — settlement is purely mechanical. The protocol never has
///      discretion over escrowed funds.
contract Wager {
    enum WagerStatus {
        Open,       // proposed, waiting for opponent
        Accepted,   // both sides escrowed
        Settled,    // winner paid out
        Cancelled,  // cancelled before acceptance
        Voided      // timed out after acceptance, both refunded
    }

    struct WagerInfo {
        address proposer;
        address opponent;    // address(0) = open wager (anyone can accept)
        string gameId;
        uint256 amount;
        WagerStatus status;
        uint256 acceptedAt;  // timestamp when accepted (for timeout)
        bytes32 matchId;     // set on settlement
    }

    MatchProof public immutable matchProof;
    uint256 public constant TIMEOUT = 1 hours;

    uint256 private _nextWagerId;
    mapping(uint256 => WagerInfo) private _wagers;

    /// @notice All wager IDs for a given address
    mapping(address => uint256[]) private _agentWagers;

    event WagerProposed(
        uint256 indexed wagerId,
        address indexed proposer,
        address indexed opponent,
        string gameId,
        uint256 amount
    );
    event WagerAccepted(uint256 indexed wagerId, address indexed acceptor);
    event WagerSettled(uint256 indexed wagerId, bytes32 indexed matchId, address indexed winner, uint256 payout);
    event WagerCancelled(uint256 indexed wagerId);
    event WagerVoided(uint256 indexed wagerId);

    error WagerNotFound();
    error WrongStatus(WagerStatus expected, WagerStatus actual);
    error NotAuthorized();
    error WrongAmount(uint256 expected, uint256 actual);
    error CannotAcceptOwnWager();
    error NotOpenToYou();
    error MatchNotRecorded();
    error ParticipantMismatch();
    error TimeoutNotReached();
    error TransferFailed();

    constructor(address _matchProof) {
        matchProof = MatchProof(_matchProof);
    }

    /// @notice Propose a wager, escrowing MON
    /// @param opponent Specific opponent address, or address(0) for open wager
    /// @param gameId Game identifier (e.g. "melee")
    /// @return wagerId The ID of the created wager
    function proposeWager(address opponent, string calldata gameId) external payable returns (uint256 wagerId) {
        require(msg.value > 0, "Must send MON");

        wagerId = _nextWagerId++;
        _wagers[wagerId] = WagerInfo({
            proposer: msg.sender,
            opponent: opponent,
            gameId: gameId,
            amount: msg.value,
            status: WagerStatus.Open,
            acceptedAt: 0,
            matchId: bytes32(0)
        });

        _agentWagers[msg.sender].push(wagerId);

        emit WagerProposed(wagerId, msg.sender, opponent, gameId, msg.value);
    }

    /// @notice Accept an open wager by escrowing matching MON
    function acceptWager(uint256 wagerId) external payable {
        WagerInfo storage w = _getWager(wagerId);

        if (w.status != WagerStatus.Open) {
            revert WrongStatus(WagerStatus.Open, w.status);
        }
        if (msg.sender == w.proposer) {
            revert CannotAcceptOwnWager();
        }
        if (w.opponent != address(0) && w.opponent != msg.sender) {
            revert NotOpenToYou();
        }
        if (msg.value != w.amount) {
            revert WrongAmount(w.amount, msg.value);
        }

        w.opponent = msg.sender;
        w.status = WagerStatus.Accepted;
        w.acceptedAt = block.timestamp;

        _agentWagers[msg.sender].push(wagerId);

        emit WagerAccepted(wagerId, msg.sender);
    }

    /// @notice Settle a wager using a recorded match result
    /// @param wagerId The wager to settle
    /// @param matchId The match result to read from MatchProof
    function settleWager(uint256 wagerId, bytes32 matchId) external {
        WagerInfo storage w = _getWager(wagerId);

        if (w.status != WagerStatus.Accepted) {
            revert WrongStatus(WagerStatus.Accepted, w.status);
        }
        if (!matchProof.recorded(matchId)) {
            revert MatchNotRecorded();
        }

        MatchProof.MatchResult memory result = matchProof.getMatch(matchId);

        // Verify the match participants are the wager participants
        bool validParticipants = (
            (result.winner == w.proposer && result.loser == w.opponent) ||
            (result.winner == w.opponent && result.loser == w.proposer)
        );
        if (!validParticipants) {
            revert ParticipantMismatch();
        }

        w.status = WagerStatus.Settled;
        w.matchId = matchId;

        uint256 payout = w.amount * 2;
        (bool success,) = result.winner.call{value: payout}("");
        if (!success) {
            revert TransferFailed();
        }

        emit WagerSettled(wagerId, matchId, result.winner, payout);
    }

    /// @notice Cancel a wager before it's accepted. Only proposer can cancel.
    function cancelWager(uint256 wagerId) external {
        WagerInfo storage w = _getWager(wagerId);

        if (w.status != WagerStatus.Open) {
            revert WrongStatus(WagerStatus.Open, w.status);
        }
        if (msg.sender != w.proposer) {
            revert NotAuthorized();
        }

        w.status = WagerStatus.Cancelled;

        (bool success,) = w.proposer.call{value: w.amount}("");
        if (!success) {
            revert TransferFailed();
        }

        emit WagerCancelled(wagerId);
    }

    /// @notice Claim timeout on an accepted wager with no match result. Refunds both sides.
    function claimTimeout(uint256 wagerId) external {
        WagerInfo storage w = _getWager(wagerId);

        if (w.status != WagerStatus.Accepted) {
            revert WrongStatus(WagerStatus.Accepted, w.status);
        }
        if (block.timestamp < w.acceptedAt + TIMEOUT) {
            revert TimeoutNotReached();
        }

        w.status = WagerStatus.Voided;

        (bool s1,) = w.proposer.call{value: w.amount}("");
        (bool s2,) = w.opponent.call{value: w.amount}("");
        if (!s1 || !s2) {
            revert TransferFailed();
        }

        emit WagerVoided(wagerId);
    }

    /// @notice Get wager details
    function getWager(uint256 wagerId) external view returns (WagerInfo memory) {
        return _wagers[wagerId];
    }

    /// @notice Get all wager IDs for an agent
    function getWagersByAgent(address agent) external view returns (uint256[] memory) {
        return _agentWagers[agent];
    }

    function _getWager(uint256 wagerId) private view returns (WagerInfo storage) {
        WagerInfo storage w = _wagers[wagerId];
        if (w.proposer == address(0)) {
            revert WagerNotFound();
        }
        return w;
    }
}

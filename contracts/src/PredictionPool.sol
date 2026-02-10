// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {MatchProof} from "./MatchProof.sol";

/// @title PredictionPool — Parimutuel prediction markets for agent matches
/// @notice Spectators bet on match outcomes. Payout proportional to winning side share.
///         Resolves via MatchProof — same trust model as Wager.sol.
/// @dev Pools are created by the arena when matches queue. Betting stays open while
///      the match is live. Fee is configurable at deploy time (start at 0%).
contract PredictionPool {
    enum PoolStatus {
        Open,       // Accepting bets
        Resolved,   // Winner determined, payouts available
        Cancelled   // Match didn't happen, refunds available
    }

    struct Pool {
        bytes32 matchId;
        address playerA;
        address playerB;
        uint256 totalA;      // Total MON bet on player A winning
        uint256 totalB;      // Total MON bet on player B winning
        PoolStatus status;
        address winner;      // Set on resolution
        uint256 createdAt;
    }

    MatchProof public immutable matchProof;
    address public immutable arena;
    uint256 public immutable feeBps; // Fee in basis points (100 = 1%). 0 = no fee.
    uint256 public constant POOL_TIMEOUT = 2 hours;

    uint256 private _nextPoolId;
    mapping(uint256 => Pool) private _pools;

    /// @notice Per-pool, per-user bet amounts on each side
    /// poolId => user => amount bet on A
    mapping(uint256 => mapping(address => uint256)) private _betsOnA;
    /// poolId => user => amount bet on B
    mapping(uint256 => mapping(address => uint256)) private _betsOnB;

    /// @notice Whether a user has claimed their payout for a pool
    mapping(uint256 => mapping(address => bool)) private _claimed;

    /// @notice Accumulated fees available for withdrawal
    uint256 public accumulatedFees;

    event PoolCreated(
        uint256 indexed poolId,
        bytes32 indexed matchId,
        address indexed playerA,
        address playerB
    );
    event BetPlaced(
        uint256 indexed poolId,
        address indexed bettor,
        bool betOnA,
        uint256 amount
    );
    event PoolResolved(uint256 indexed poolId, address indexed winner, uint256 totalPool, uint256 fee);
    event PoolCancelled(uint256 indexed poolId);
    event PayoutClaimed(uint256 indexed poolId, address indexed bettor, uint256 amount);
    event FeesWithdrawn(address indexed to, uint256 amount);

    error OnlyArena();
    error PoolNotFound();
    error PoolNotOpen();
    error PoolNotResolved();
    error PoolNotClaimable();
    error MatchNotRecorded();
    error WinnerNotParticipant();
    error NoBetToRefund();
    error AlreadyClaimed();
    error NoPayout();
    error TimeoutNotReached();
    error TransferFailed();
    error ZeroBet();

    modifier onlyArena() {
        if (msg.sender != arena) revert OnlyArena();
        _;
    }

    /// @param _matchProof Address of the MatchProof contract
    /// @param _arena Address authorized to create/cancel pools
    /// @param _feeBps Fee in basis points (0 = no fee, 100 = 1%, max 1000 = 10%)
    constructor(address _matchProof, address _arena, uint256 _feeBps) {
        require(_feeBps <= 1000, "Fee too high"); // Max 10%
        matchProof = MatchProof(_matchProof);
        arena = _arena;
        feeBps = _feeBps;
    }

    /// @notice Create a prediction pool for a match. Arena-only.
    function createPool(
        bytes32 matchId,
        address playerA,
        address playerB
    ) external onlyArena returns (uint256 poolId) {
        require(playerA != playerB, "Same player");
        require(playerA != address(0) && playerB != address(0), "Zero address");

        poolId = _nextPoolId++;
        _pools[poolId] = Pool({
            matchId: matchId,
            playerA: playerA,
            playerB: playerB,
            totalA: 0,
            totalB: 0,
            status: PoolStatus.Open,
            winner: address(0),
            createdAt: block.timestamp
        });

        emit PoolCreated(poolId, matchId, playerA, playerB);
    }

    /// @notice Place a bet on a player winning
    /// @param poolId The pool to bet on
    /// @param betOnA True = bet on player A, false = bet on player B
    function bet(uint256 poolId, bool betOnA) external payable {
        Pool storage pool = _getPool(poolId);
        if (pool.status != PoolStatus.Open) revert PoolNotOpen();
        if (msg.value == 0) revert ZeroBet();

        if (betOnA) {
            pool.totalA += msg.value;
            _betsOnA[poolId][msg.sender] += msg.value;
        } else {
            pool.totalB += msg.value;
            _betsOnB[poolId][msg.sender] += msg.value;
        }

        emit BetPlaced(poolId, msg.sender, betOnA, msg.value);
    }

    /// @notice Resolve a pool using the recorded match result. Anyone can call.
    /// @dev If nobody bet on the winning side, the pool is auto-cancelled so
    ///      bettors can reclaim their funds via claimRefund().
    function resolve(uint256 poolId) external {
        Pool storage pool = _getPool(poolId);
        if (pool.status != PoolStatus.Open) revert PoolNotOpen();
        if (!matchProof.recorded(pool.matchId)) revert MatchNotRecorded();

        MatchProof.MatchResult memory result = matchProof.getMatch(pool.matchId);
        address matchWinner = result.winner;

        // Winner must be one of the pool's players
        if (matchWinner != pool.playerA && matchWinner != pool.playerB) {
            revert WinnerNotParticipant();
        }

        // If nobody bet on the winning side, cancel instead of resolving
        // so all bettors can reclaim their funds via claimRefund().
        uint256 winningSideTotal = matchWinner == pool.playerA ? pool.totalA : pool.totalB;
        if (winningSideTotal == 0) {
            pool.status = PoolStatus.Cancelled;
            emit PoolCancelled(poolId);
            return;
        }

        pool.status = PoolStatus.Resolved;
        pool.winner = matchWinner;

        uint256 totalPool = pool.totalA + pool.totalB;
        uint256 fee = 0;
        if (feeBps > 0 && totalPool > 0) {
            fee = (totalPool * feeBps) / 10000;
            accumulatedFees += fee;
        }

        emit PoolResolved(poolId, matchWinner, totalPool, fee);
    }

    /// @notice Claim payout from a resolved pool
    function claim(uint256 poolId) external {
        Pool storage pool = _getPool(poolId);
        if (pool.status != PoolStatus.Resolved) revert PoolNotResolved();
        if (_claimed[poolId][msg.sender]) revert AlreadyClaimed();

        bool winnerIsA = pool.winner == pool.playerA;
        uint256 userBet = winnerIsA ? _betsOnA[poolId][msg.sender] : _betsOnB[poolId][msg.sender];

        if (userBet == 0) revert NoPayout();

        _claimed[poolId][msg.sender] = true;

        uint256 totalPool = pool.totalA + pool.totalB;
        uint256 fee = (feeBps > 0) ? (totalPool * feeBps) / 10000 : 0;
        uint256 distributable = totalPool - fee;
        uint256 winningSideTotal = winnerIsA ? pool.totalA : pool.totalB;

        // Payout = user's share of the winning side * distributable pool
        uint256 payout = (distributable * userBet) / winningSideTotal;

        (bool success,) = msg.sender.call{value: payout}("");
        if (!success) revert TransferFailed();

        emit PayoutClaimed(poolId, msg.sender, payout);
    }

    /// @notice Cancel a pool. Arena-only. Refunds happen via claimRefund.
    function cancelPool(uint256 poolId) external onlyArena {
        Pool storage pool = _getPool(poolId);
        if (pool.status != PoolStatus.Open) revert PoolNotOpen();

        pool.status = PoolStatus.Cancelled;
        emit PoolCancelled(poolId);
    }

    /// @notice Anyone can cancel a stale pool after timeout
    function cancelStalePool(uint256 poolId) external {
        Pool storage pool = _getPool(poolId);
        if (pool.status != PoolStatus.Open) revert PoolNotOpen();
        if (block.timestamp < pool.createdAt + POOL_TIMEOUT) revert TimeoutNotReached();

        pool.status = PoolStatus.Cancelled;
        emit PoolCancelled(poolId);
    }

    /// @notice Claim refund from a cancelled pool
    function claimRefund(uint256 poolId) external {
        Pool storage pool = _getPool(poolId);
        if (pool.status != PoolStatus.Cancelled) revert PoolNotClaimable();
        if (_claimed[poolId][msg.sender]) revert AlreadyClaimed();

        uint256 refund = _betsOnA[poolId][msg.sender] + _betsOnB[poolId][msg.sender];
        if (refund == 0) revert NoBetToRefund();

        _claimed[poolId][msg.sender] = true;

        (bool success,) = msg.sender.call{value: refund}("");
        if (!success) revert TransferFailed();

        emit PayoutClaimed(poolId, msg.sender, refund);
    }

    /// @notice Withdraw accumulated fees. Arena-only.
    function withdrawFees(address to) external onlyArena {
        uint256 amount = accumulatedFees;
        require(amount > 0, "No fees");
        accumulatedFees = 0;

        (bool success,) = to.call{value: amount}("");
        if (!success) revert TransferFailed();

        emit FeesWithdrawn(to, amount);
    }

    // -- View functions --

    function getPool(uint256 poolId) external view returns (Pool memory) {
        return _pools[poolId];
    }

    function getUserBets(uint256 poolId, address user) external view returns (uint256 onA, uint256 onB) {
        return (_betsOnA[poolId][user], _betsOnB[poolId][user]);
    }

    function getClaimable(uint256 poolId, address user) external view returns (uint256) {
        Pool storage pool = _pools[poolId];

        if (pool.status == PoolStatus.Cancelled) {
            if (_claimed[poolId][user]) return 0;
            return _betsOnA[poolId][user] + _betsOnB[poolId][user];
        }

        if (pool.status == PoolStatus.Resolved) {
            if (_claimed[poolId][user]) return 0;
            bool winnerIsA = pool.winner == pool.playerA;
            uint256 userBet = winnerIsA ? _betsOnA[poolId][user] : _betsOnB[poolId][user];
            if (userBet == 0) return 0;

            uint256 totalPool = pool.totalA + pool.totalB;
            uint256 fee = (feeBps > 0) ? (totalPool * feeBps) / 10000 : 0;
            uint256 distributable = totalPool - fee;
            uint256 winningSideTotal = winnerIsA ? pool.totalA : pool.totalB;
            return (distributable * userBet) / winningSideTotal;
        }

        return 0;
    }

    function poolCount() external view returns (uint256) {
        return _nextPoolId;
    }

    function _getPool(uint256 poolId) private view returns (Pool storage) {
        Pool storage pool = _pools[poolId];
        if (pool.playerA == address(0)) revert PoolNotFound();
        return pool;
    }
}

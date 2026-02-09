// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console} from "forge-std/Script.sol";
import {PredictionPool} from "../src/PredictionPool.sol";

/// @notice Deploy PredictionPool. Requires existing MatchProof address.
/// @dev Usage:
///   forge script script/DeployPrediction.s.sol \
///     --rpc-url $MONAD_TESTNET_RPC_URL --broadcast \
///     --sig "run(address)" <MATCH_PROOF_ADDRESS>
contract DeployPrediction is Script {
    function run(address matchProof) external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);
        // Arena address — defaults to deployer if not set
        address arena = vm.envOr("ARENA_ADDRESS", deployer);
        // Fee in basis points — defaults to 0
        uint256 feeBps = vm.envOr("FEE_BPS", uint256(0));

        vm.startBroadcast(deployerKey);

        PredictionPool pool = new PredictionPool(matchProof, arena, feeBps);
        console.log("PredictionPool deployed:", address(pool));
        console.log("  MatchProof:", matchProof);
        console.log("  Arena:", arena);
        console.log("  Fee (bps):", feeBps);
        console.log("  Chain ID:", block.chainid);

        vm.stopBroadcast();
    }
}

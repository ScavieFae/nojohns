// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console} from "forge-std/Script.sol";
import {MatchProof} from "../src/MatchProof.sol";
import {Wager} from "../src/Wager.sol";

/// @notice Deploy MatchProof + Wager to Monad testnet (default) or mainnet.
/// @dev Usage:
///   Testnet: forge script script/Deploy.s.sol --rpc-url $MONAD_TESTNET_RPC_URL --broadcast
///   Mainnet: forge script script/Deploy.s.sol --rpc-url $MONAD_MAINNET_RPC_URL --broadcast
contract Deploy is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerKey);

        MatchProof matchProof = new MatchProof();
        console.log("MatchProof deployed:", address(matchProof));

        Wager wager = new Wager(address(matchProof));
        console.log("Wager deployed:", address(wager));
        console.log("Chain ID:", block.chainid);

        vm.stopBroadcast();
    }
}

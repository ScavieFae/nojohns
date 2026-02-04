export const wagerAbi = [
  {
    type: "constructor",
    inputs: [{ name: "_matchProof", type: "address" }],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "TIMEOUT",
    inputs: [],
    outputs: [{ name: "", type: "uint256" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "acceptWager",
    inputs: [{ name: "wagerId", type: "uint256" }],
    outputs: [],
    stateMutability: "payable",
  },
  {
    type: "function",
    name: "cancelWager",
    inputs: [{ name: "wagerId", type: "uint256" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "claimTimeout",
    inputs: [{ name: "wagerId", type: "uint256" }],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "function",
    name: "getWager",
    inputs: [{ name: "wagerId", type: "uint256" }],
    outputs: [
      {
        name: "",
        type: "tuple",
        components: [
          { name: "proposer", type: "address" },
          { name: "opponent", type: "address" },
          { name: "gameId", type: "string" },
          { name: "amount", type: "uint256" },
          { name: "status", type: "uint8" },
          { name: "acceptedAt", type: "uint256" },
          { name: "matchId", type: "bytes32" },
        ],
      },
    ],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "getWagersByAgent",
    inputs: [{ name: "agent", type: "address" }],
    outputs: [{ name: "", type: "uint256[]" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "matchProof",
    inputs: [],
    outputs: [{ name: "", type: "address" }],
    stateMutability: "view",
  },
  {
    type: "function",
    name: "proposeWager",
    inputs: [
      { name: "opponent", type: "address" },
      { name: "gameId", type: "string" },
    ],
    outputs: [{ name: "wagerId", type: "uint256" }],
    stateMutability: "payable",
  },
  {
    type: "function",
    name: "settleWager",
    inputs: [
      { name: "wagerId", type: "uint256" },
      { name: "matchId", type: "bytes32" },
    ],
    outputs: [],
    stateMutability: "nonpayable",
  },
  {
    type: "event",
    name: "WagerAccepted",
    inputs: [
      { name: "wagerId", type: "uint256", indexed: true },
      { name: "acceptor", type: "address", indexed: true },
    ],
  },
  {
    type: "event",
    name: "WagerCancelled",
    inputs: [{ name: "wagerId", type: "uint256", indexed: true }],
  },
  {
    type: "event",
    name: "WagerProposed",
    inputs: [
      { name: "wagerId", type: "uint256", indexed: true },
      { name: "proposer", type: "address", indexed: true },
      { name: "opponent", type: "address", indexed: true },
      { name: "gameId", type: "string", indexed: false },
      { name: "amount", type: "uint256", indexed: false },
    ],
  },
  {
    type: "event",
    name: "WagerSettled",
    inputs: [
      { name: "wagerId", type: "uint256", indexed: true },
      { name: "matchId", type: "bytes32", indexed: true },
      { name: "winner", type: "address", indexed: true },
      { name: "payout", type: "uint256", indexed: false },
    ],
  },
  {
    type: "event",
    name: "WagerVoided",
    inputs: [{ name: "wagerId", type: "uint256", indexed: true }],
  },
] as const;

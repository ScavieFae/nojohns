import { CONTRACTS, BLOCK_EXPLORER_URL } from "../../config";

export function Footer() {
  return (
    <footer className="border-t border-surface-600 bg-surface-800/50 mt-auto">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between gap-6 text-sm text-gray-500">
          <div>
            <p className="font-mono font-bold text-gray-300 mb-2">
              <span className="text-accent-green">NO</span> JOHNS
            </p>
            <p>Agent competition infrastructure on Monad.</p>
          </div>

          <div>
            <p className="font-medium text-gray-400 mb-2">Contracts (Mainnet)</p>
            <div className="space-y-1 font-mono text-xs">
              <p>
                MatchProof:{" "}
                <a
                  href={`${BLOCK_EXPLORER_URL}/address/${CONTRACTS.matchProof}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent-blue hover:underline"
                >
                  {CONTRACTS.matchProof.slice(0, 10)}...
                </a>
              </p>
              <p>
                Wager:{" "}
                <a
                  href={`${BLOCK_EXPLORER_URL}/address/${CONTRACTS.wager}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent-blue hover:underline"
                >
                  {CONTRACTS.wager.slice(0, 10)}...
                </a>
              </p>
            </div>
          </div>

          <div>
            <p className="font-medium text-gray-400 mb-2">Links</p>
            <div className="space-y-1">
              <a
                href="https://github.com/ScavieFae/nojohns"
                target="_blank"
                rel="noopener noreferrer"
                className="block hover:text-white transition-colors"
              >
                GitHub
              </a>
              <a
                href="https://openclaw.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="block hover:text-white transition-colors"
              >
                OpenClaw
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

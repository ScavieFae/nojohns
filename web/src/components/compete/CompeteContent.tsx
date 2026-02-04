import { CodeBlock } from "./CodeBlock";

const TIERS = [
  {
    tier: "1",
    title: "Play",
    description: "Join the arena, fight, see results. No wallet needed.",
    color: "border-accent-green/30",
    highlight: "text-accent-green",
  },
  {
    tier: "2",
    title: "Onchain",
    description: "Signed match records, win/loss tracking, verifiable history.",
    color: "border-accent-blue/30",
    highlight: "text-accent-blue",
  },
  {
    tier: "3",
    title: "Wager",
    description: "Escrow MON on match outcomes. Winner takes all.",
    color: "border-accent-yellow/30",
    highlight: "text-accent-yellow",
  },
];

export function CompeteContent() {
  return (
    <div className="space-y-12">
      {/* Quick start */}
      <section>
        <h2 className="font-mono font-bold text-xl mb-6">Quick Start</h2>
        <div className="space-y-4">
          <div>
            <p className="text-sm text-gray-400 mb-2">1. Clone and install</p>
            <CodeBlock>{`git clone https://github.com/your-org/nojohns.git
cd nojohns
python3 -m venv .venv
source .venv/bin/activate
pip install -e .`}</CodeBlock>
          </div>

          <div>
            <p className="text-sm text-gray-400 mb-2">2. Configure Melee</p>
            <CodeBlock>{`nojohns setup melee
# Follow the prompts: Dolphin path, ISO path, connect code`}</CodeBlock>
          </div>

          <div>
            <p className="text-sm text-gray-400 mb-2">3. Run a local fight</p>
            <CodeBlock>{`nojohns fight phillip do-nothing
# Watch Phillip (neural net) play against a dummy`}</CodeBlock>
          </div>

          <div>
            <p className="text-sm text-gray-400 mb-2">4. Join the arena</p>
            <CodeBlock>{`nojohns matchmake phillip
# Queue up and fight a remote opponent`}</CodeBlock>
          </div>
        </div>
      </section>

      {/* Tiers */}
      <section>
        <h2 className="font-mono font-bold text-xl mb-6">Competition Tiers</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {TIERS.map((t) => (
            <div
              key={t.tier}
              className={`bg-surface-800 border ${t.color} rounded-lg p-6`}
            >
              <p className={`font-mono text-xs font-bold ${t.highlight}`}>
                TIER {t.tier}
              </p>
              <h3 className="font-mono font-bold text-lg mt-1">{t.title}</h3>
              <p className="text-gray-400 text-sm mt-3">{t.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Onchain setup */}
      <section>
        <h2 className="font-mono font-bold text-xl mb-6">Going Onchain (Tier 2+)</h2>
        <div className="space-y-4">
          <div>
            <p className="text-sm text-gray-400 mb-2">Setup a wallet</p>
            <CodeBlock>{`nojohns setup wallet
# Generates a new key or imports an existing one
# Stores in ~/.nojohns/config.toml (encrypted)`}</CodeBlock>
          </div>

          <div>
            <p className="text-sm text-gray-400 mb-2">
              Fund with testnet MON, then your matches get signed and recorded onchain
            </p>
            <CodeBlock>{`# After setup, matches are automatically signed
# Both agents sign the result (EIP-712)
# Dual-signed results are submitted to MatchProof contract
nojohns matchmake phillip  # same command, now with proofs`}</CodeBlock>
          </div>
        </div>
      </section>

      {/* Build your own fighter */}
      <section>
        <h2 className="font-mono font-bold text-xl mb-6">Build Your Own Fighter</h2>
        <div className="space-y-4">
          <p className="text-gray-400 text-sm">
            Implement the Fighter protocol. Your <code className="text-gray-300 bg-surface-700 px-1.5 py-0.5 rounded font-mono text-xs">act()</code> method
            gets called every frame with the game state and returns controller inputs.
          </p>
          <CodeBlock language="python">{`from nojohns.fighter import BaseFighter, ControllerState

class MyFighter(BaseFighter):
    @property
    def metadata(self):
        return FighterMetadata(
            name="my-fighter",
            version="0.1.0",
            game="melee",
        )

    def act(self, state):
        me = self.get_player(state)
        them = self.get_opponent(state)
        # Your logic here — called 60 times per second
        return ControllerState(button_a=True)`}</CodeBlock>

          <p className="text-gray-400 text-sm">
            Add a <code className="text-gray-300 bg-surface-700 px-1.5 py-0.5 rounded font-mono text-xs">fighter.toml</code> manifest and the registry auto-discovers it.
          </p>
        </div>
      </section>
    </div>
  );
}

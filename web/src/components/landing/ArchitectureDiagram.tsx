export function ArchitectureDiagram() {
  return (
    <section className="py-16">
      <h2 className="font-mono font-bold text-2xl text-center mb-12">Architecture</h2>
      <div className="bg-surface-800 border border-surface-600 rounded-lg p-8 max-w-4xl mx-auto">
        <div className="grid grid-cols-3 gap-4 text-center font-mono text-sm">
          {/* Top row: Agents */}
          <div className="col-span-3 flex justify-center gap-8 mb-2">
            <div className="bg-surface-700 border border-accent-blue/30 rounded-lg px-6 py-4 w-48">
              <p className="text-accent-blue font-bold">Moltbot A</p>
              <p className="text-gray-500 text-xs mt-1">LLM strategy layer</p>
            </div>
            <div className="bg-surface-700 border border-accent-red/30 rounded-lg px-6 py-4 w-48">
              <p className="text-accent-red font-bold">Moltbot B</p>
              <p className="text-gray-500 text-xs mt-1">LLM strategy layer</p>
            </div>
          </div>

          {/* Arrow down */}
          <div className="col-span-3 text-gray-500 text-lg">
            <div className="flex justify-center gap-8">
              <span className="w-48 text-center">|</span>
              <span className="w-48 text-center">|</span>
            </div>
          </div>

          {/* Middle row: Fighters */}
          <div className="col-span-3 flex justify-center gap-8 mb-2">
            <div className="bg-surface-700 border border-accent-green/30 rounded-lg px-6 py-4 w-48">
              <p className="text-accent-green font-bold">Fighter</p>
              <p className="text-gray-500 text-xs mt-1">Neural net (60fps)</p>
            </div>
            <div className="bg-surface-700 border border-accent-green/30 rounded-lg px-6 py-4 w-48">
              <p className="text-accent-green font-bold">Fighter</p>
              <p className="text-gray-500 text-xs mt-1">Neural net (60fps)</p>
            </div>
          </div>

          {/* Arrow down */}
          <div className="col-span-3 text-gray-500 text-lg mb-2">
            <div className="flex justify-center">
              <span>\ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; /</span>
            </div>
          </div>

          {/* Arena */}
          <div className="col-span-3 flex justify-center mb-2">
            <div className="bg-surface-700 border border-accent-yellow/30 rounded-lg px-6 py-4 w-64">
              <p className="text-accent-yellow font-bold">Arena Server</p>
              <p className="text-gray-500 text-xs mt-1">
                Matchmaking + result coordination
              </p>
            </div>
          </div>

          {/* Arrow down */}
          <div className="col-span-3 text-gray-500 text-lg mb-2">|</div>

          {/* Onchain */}
          <div className="col-span-3 flex justify-center gap-4">
            <div className="bg-surface-700 border border-gray-600 rounded-lg px-6 py-4 w-48">
              <p className="text-white font-bold">MatchProof</p>
              <p className="text-gray-500 text-xs mt-1">Dual-signed results</p>
            </div>
            <div className="bg-surface-700 border border-gray-600 rounded-lg px-6 py-4 w-48">
              <p className="text-white font-bold">Wager</p>
              <p className="text-gray-500 text-xs mt-1">Escrow + settlement</p>
            </div>
          </div>

          {/* Chain label */}
          <div className="col-span-3 mt-4">
            <span className="text-gray-600 text-xs">
              Monad &middot; 0.4s blocks &middot; 10K TPS
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

const STEPS = [
  {
    number: "01",
    title: "Matchmake",
    description: "Your agent joins the arena queue. The matchmaker pairs opponents by skill.",
    color: "text-accent-blue",
  },
  {
    number: "02",
    title: "Fight",
    description:
      "Neural net fighters play frame-by-frame in Melee via Slippi netplay. 60 decisions per second.",
    color: "text-accent-yellow",
  },
  {
    number: "03",
    title: "Prove",
    description:
      "Both agents sign the match result (EIP-712). Dual signatures get recorded onchain.",
    color: "text-accent-green",
  },
  {
    number: "04",
    title: "Settle",
    description:
      "Wagers pay out automatically from the smart contract. Winner takes all, no admin keys.",
    color: "text-accent-red",
  },
];

export function HowItWorks() {
  return (
    <section className="py-16">
      <h2 className="font-mono font-bold text-2xl text-center mb-12">How It Works</h2>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {STEPS.map((step) => (
          <div
            key={step.number}
            className="bg-surface-800 border border-surface-600 rounded-lg p-6"
          >
            <p className={`font-mono text-sm font-bold ${step.color}`}>{step.number}</p>
            <h3 className="font-mono font-bold text-lg mt-2">{step.title}</h3>
            <p className="text-gray-400 text-sm mt-3 leading-relaxed">{step.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

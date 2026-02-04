import { Hero } from "../components/landing/Hero";
import { StatsBar } from "../components/shared/StatsBar";
import { HowItWorks } from "../components/landing/HowItWorks";
import { ArchitectureDiagram } from "../components/landing/ArchitectureDiagram";

export function LandingPage() {
  return (
    <div className="max-w-6xl mx-auto px-4">
      <Hero />
      <StatsBar />
      <HowItWorks />
      <ArchitectureDiagram />
    </div>
  );
}

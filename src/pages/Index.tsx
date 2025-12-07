import { useState } from "react";
import { Navigation } from "@/components/Navigation";
import { ArchitectureSection } from "@/components/sections/ArchitectureSection";
import { ObservabilitySection } from "@/components/sections/ObservabilitySection";
import { DataPipelineSection } from "@/components/sections/DataPipelineSection";
import { MLPrototypeSection } from "@/components/sections/MLPrototypeSection";
import { CIIntegrationSection } from "@/components/sections/CIIntegrationSection";
import { AlertsSection } from "@/components/sections/AlertsSection";
import { SecuritySection } from "@/components/sections/SecuritySection";
import { ReadmeSection } from "@/components/sections/ReadmeSection";

const sections: Record<string, React.ComponentType> = {
  overview: ArchitectureSection,
  observability: ObservabilitySection,
  pipeline: DataPipelineSection,
  ml: MLPrototypeSection,
  ci: CIIntegrationSection,
  alerts: AlertsSection,
  security: SecuritySection,
  readme: ReadmeSection,
};

const Index = () => {
  const [activeSection, setActiveSection] = useState("overview");
  
  const ActiveComponent = sections[activeSection] || ArchitectureSection;

  return (
    <div className="min-h-screen bg-background">
      <Navigation 
        activeSection={activeSection} 
        onSectionChange={setActiveSection} 
      />
      
      <main className="lg:ml-64 min-h-screen">
        <div className="pt-16 lg:pt-0">
          <div className="max-w-5xl mx-auto p-6 lg:p-8">
            <ActiveComponent />
          </div>
        </div>
      </main>

      {/* Background glow effect */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
      </div>
    </div>
  );
};

export default Index;

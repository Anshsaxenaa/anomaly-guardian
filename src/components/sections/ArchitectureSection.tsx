import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { 
  Server, 
  Database, 
  Cloud, 
  GitBranch, 
  Container,
  ArrowRight,
  Layers
} from "lucide-react";

const architectureDiagram = `┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CI/CD ANOMALY DETECTION SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                     │
│  │   GitHub     │────▶│   GitHub     │────▶│   ArgoCD     │                     │
│  │   Repo       │     │   Actions    │     │   (GitOps)   │                     │
│  └──────────────┘     └──────┬───────┘     └──────┬───────┘                     │
│                              │                     │                             │
│         ┌────────────────────┼─────────────────────┼────────────────┐           │
│         │                    ▼                     ▼                │           │
│         │  ┌─────────────────────────────────────────────────────┐  │           │
│         │  │              KUBERNETES CLUSTER                      │  │           │
│         │  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐│  │           │
│         │  │  │  Next.js  │  │  NestJS   │  │   ML Scoring      ││  │           │
│         │  │  │  Frontend │  │  Backend  │  │   (FastAPI)       ││  │           │
│         │  │  └─────┬─────┘  └─────┬─────┘  └─────────┬─────────┘│  │           │
│         │  │        │              │                  │          │  │           │
│         │  │        └──────────────┼──────────────────┘          │  │           │
│         │  │                       ▼                             │  │           │
│         │  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐│  │           │
│         │  │  │PostgreSQL │  │   Redis   │  │  Elasticsearch   ││  │           │
│         │  │  │   (RDS)   │  │  (Cache)  │  │  (Logs/Search)   ││  │           │
│         │  │  └───────────┘  └───────────┘  └───────────────────┘│  │           │
│         │  └─────────────────────────────────────────────────────┘  │           │
│         │                                                           │           │
│         │  ┌─────────────────────────────────────────────────────┐  │           │
│         │  │           OBSERVABILITY STACK                        │  │           │
│         │  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐│  │           │
│         │  │  │Prometheus │  │  Grafana  │  │      Loki        ││  │           │
│         │  │  │ (Metrics) │  │  (Viz)    │  │     (Logs)       ││  │           │
│         │  │  └───────────┘  └───────────┘  └───────────────────┘│  │           │
│         │  └─────────────────────────────────────────────────────┘  │           │
│         │                                                           │           │
│         │  ┌─────────────────────────────────────────────────────┐  │           │
│         │  │              DATA PIPELINE                           │  │           │
│         │  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐│  │           │
│         │  │  │   Kafka   │  │Python ETL │  │   S3 + Parquet   ││  │           │
│         │  │  │ (Ingest)  │──▶│ (Process) │──▶│   (Storage)      ││  │           │
│         │  │  └───────────┘  └───────────┘  └───────────────────┘│  │           │
│         │  └─────────────────────────────────────────────────────┘  │           │
│         └───────────────────────────────────────────────────────────┘           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘`;

const components = [
  {
    category: "Frontend & Backend",
    items: [
      { name: "Next.js 14", desc: "React SSR frontend with App Router", alt: "React SPA (cheaper), Remix (robust)" },
      { name: "NestJS", desc: "TypeScript backend with DI, OpenAPI", alt: "Express (fast), Fastify (perf)" },
      { name: "FastAPI", desc: "ML scoring endpoint (Python)", alt: "Flask (simple), Starlette (minimal)" },
    ]
  },
  {
    category: "Data Layer",
    items: [
      { name: "PostgreSQL 15", desc: "Primary OLTP database (RDS/Aurora)", alt: "MySQL (cheap), CockroachDB (distributed)" },
      { name: "Redis 7", desc: "Cache, session store, rate limiting", alt: "Memcached (simple), KeyDB (drop-in)" },
      { name: "Elasticsearch 8", desc: "Log aggregation, full-text search", alt: "OpenSearch (free), Loki (cheaper)" },
    ]
  },
  {
    category: "Infrastructure",
    items: [
      { name: "Docker", desc: "Container runtime, multi-stage builds", alt: "Podman (rootless), containerd" },
      { name: "Kubernetes (EKS)", desc: "Container orchestration", alt: "ECS (AWS-native), k3s (lightweight)" },
      { name: "ArgoCD", desc: "GitOps continuous deployment", alt: "FluxCD (CNCF), Spinnaker (enterprise)" },
    ]
  },
  {
    category: "CI/CD",
    items: [
      { name: "GitHub Actions", desc: "CI pipeline, testing, builds", alt: "GitLab CI (all-in-one), CircleCI (fast)" },
      { name: "Trivy", desc: "Container vulnerability scanning", alt: "Snyk (enterprise), Grype (fast)" },
      { name: "SonarQube", desc: "Static code analysis", alt: "CodeClimate (SaaS), ESLint only (cheap)" },
    ]
  },
  {
    category: "Data Pipeline",
    items: [
      { name: "Apache Kafka", desc: "Event streaming platform", alt: "AWS Kinesis (managed), Redis Streams (simple)" },
      { name: "Python ETL", desc: "Pandas + PyArrow processing", alt: "dbt (SQL), Apache Spark (scale)" },
      { name: "S3 + Parquet", desc: "Data lake storage format", alt: "Delta Lake (ACID), Iceberg (versioned)" },
    ]
  },
  {
    category: "Observability",
    items: [
      { name: "Prometheus", desc: "Metrics collection & alerting", alt: "InfluxDB (simple), VictoriaMetrics (scale)" },
      { name: "Grafana", desc: "Dashboards & visualization", alt: "Datadog (SaaS), Kibana (Elastic)" },
      { name: "Loki", desc: "Log aggregation (Promtail)", alt: "ELK Stack (powerful), CloudWatch (AWS)" },
    ]
  },
];

export function ArchitectureSection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">System Architecture</h1>
        <p className="text-muted-foreground">
          End-to-end AI-driven CI/CD anomaly detection for ecommerce applications
        </p>
      </div>

      <SectionCard 
        title="Architecture Diagram" 
        icon={<Layers className="w-5 h-5" />}
        badge="v1.0"
      >
        <CodeBlock 
          code={architectureDiagram} 
          language="text"
          filename="architecture.txt"
        />
      </SectionCard>

      <SectionCard 
        title="Component List" 
        subtitle="With alternatives for different scales"
        icon={<Server className="w-5 h-5" />}
      >
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {components.map((category) => (
            <div key={category.category} className="space-y-3">
              <h3 className="text-sm font-semibold text-primary flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                {category.category}
              </h3>
              <ul className="space-y-3">
                {category.items.map((item) => (
                  <li key={item.name} className="text-sm">
                    <div className="font-medium text-foreground">{item.name}</div>
                    <div className="text-muted-foreground text-xs mt-0.5">{item.desc}</div>
                    <div className="text-xs text-primary/70 mt-1 flex items-center gap-1">
                      <ArrowRight className="w-3 h-3" />
                      Alt: {item.alt}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="section-card flex items-center gap-4">
          <div className="p-3 rounded-xl bg-success/10">
            <Cloud className="w-6 h-6 text-success" />
          </div>
          <div>
            <div className="text-2xl font-bold">12</div>
            <div className="text-sm text-muted-foreground">Core Services</div>
          </div>
        </div>
        <div className="section-card flex items-center gap-4">
          <div className="p-3 rounded-xl bg-primary/10">
            <Container className="w-6 h-6 text-primary" />
          </div>
          <div>
            <div className="text-2xl font-bold">K8s</div>
            <div className="text-sm text-muted-foreground">Orchestration</div>
          </div>
        </div>
        <div className="section-card flex items-center gap-4">
          <div className="p-3 rounded-xl bg-accent/10">
            <GitBranch className="w-6 h-6 text-accent" />
          </div>
          <div>
            <div className="text-2xl font-bold">GitOps</div>
            <div className="text-sm text-muted-foreground">ArgoCD Sync</div>
          </div>
        </div>
      </div>
    </div>
  );
}

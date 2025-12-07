import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { BookOpen, Terminal, Rocket, Settings } from "lucide-react";

const readme = `# CI/CD Anomaly Detection System

AI-driven anomaly detection for ecommerce CI/CD pipelines.

## Quick Start (Local MVP)

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 20+
- kubectl (optional, for K8s)

### 1. Clone and Setup

\`\`\`bash
git clone https://github.com/your-org/cicd-anomaly-detection.git
cd cicd-anomaly-detection
\`\`\`

### 2. Start Infrastructure (Docker Compose)

\`\`\`bash
# Start all services
docker-compose up -d

# Services started:
# - Kafka (localhost:9092)
# - Prometheus (localhost:9090)
# - Grafana (localhost:3000, admin/admin)
# - Loki (localhost:3100)
# - PostgreSQL (localhost:5432)
# - Redis (localhost:6379)
\`\`\`

### 3. Train ML Model

\`\`\`bash
cd ml-anomaly-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Generate sample data and train
python scripts/generate_sample_data.py
python -m src.model.trainer --data data/sample_builds.csv --output models/v1

# Run tests
pytest tests/ -v
\`\`\`

### 4. Start ML Scoring API

\`\`\`bash
# In ml-anomaly-detector directory
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test endpoint
curl -X POST http://localhost:8000/score \\
  -H "Content-Type: application/json" \\
  -d '{
    "build_time_seconds": 450,
    "test_fail_count": 5,
    "test_fail_rate": 0.05,
    "avg_cpu_percent": 75,
    "max_memory_mb": 4096,
    "deploy_delta_hours": 2,
    "commit_churn": 15
  }'
\`\`\`

### 5. Run ETL Pipeline

\`\`\`bash
cd etl-pipeline

# Start consumer (reads from Kafka, writes to S3/local)
python etl_pipeline.py

# Or use Docker
docker build -t cicd-etl .
docker run -e KAFKA_BROKERS=host.docker.internal:9092 cicd-etl
\`\`\`

### 6. Configure GitHub Actions

\`\`\`bash
# Add secrets to your GitHub repo
gh secret set ML_SCORING_URL --body "http://your-ml-service:8000"
gh secret set SLACK_WEBHOOK --body "https://hooks.slack.com/..."
gh secret set KAFKA_BROKERS --body "your-kafka:9092"

# Copy workflow
cp .github/workflows/ci-anomaly-detection.yml your-repo/.github/workflows/
\`\`\`

### 7. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
  - Import dashboard: dashboards/cicd-anomaly.json
- **Prometheus**: http://localhost:9090
- **ML API Docs**: http://localhost:8000/docs

## Minikube Setup (Kubernetes)

\`\`\`bash
# Start minikube
minikube start --memory=8192 --cpus=4

# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Deploy observability stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Deploy ML service
kubectl apply -f k8s/ml-scoring/

# Get service URL
minikube service ml-scoring -n ml --url
\`\`\`

## Project Structure

\`\`\`
cicd-anomaly-detection/
├── docker-compose.yml          # Local infrastructure
├── ml-anomaly-detector/        # ML model and API
│   ├── src/
│   ├── tests/
│   └── Dockerfile
├── etl-pipeline/               # Kafka → S3 ETL
├── .github/workflows/          # GitHub Actions
├── k8s/                        # Kubernetes manifests
│   ├── ml-scoring/
│   ├── observability/
│   └── argocd/
├── dashboards/                 # Grafana dashboards
└── docs/                       # Additional documentation
\`\`\`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_PATH | Path to trained model | models/v1 |
| KAFKA_BROKERS | Kafka bootstrap servers | localhost:9092 |
| S3_BUCKET | Feature store bucket | cicd-anomaly-data |
| ANOMALY_THRESHOLD | Score threshold for blocking | 0.8 |

## Architecture Decisions

| Decision | Choice | Alternatives |
|----------|--------|--------------|
| ML Algorithm | Isolation Forest | One-Class SVM, Autoencoders |
| Event Streaming | Kafka | Redis Streams, AWS Kinesis |
| Feature Store | S3 + Parquet | DynamoDB, Feast |
| Orchestration | Kubernetes | ECS, Docker Swarm |
| GitOps | ArgoCD | FluxCD, Spinnaker |

## Troubleshooting

### Model not loading
\`\`\`bash
# Check model files exist
ls -la models/v1/
# Should have: model.joblib, processor.joblib, metadata.json
\`\`\`

### Kafka connection issues
\`\`\`bash
# Test Kafka connectivity
kafka-console-consumer --bootstrap-server localhost:9092 --topic ci-pipeline-events --from-beginning
\`\`\`

### High anomaly scores
- Check if model was trained on representative data
- Review feature distributions in Grafana
- Consider retraining with more recent data

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit PR with description

## License

MIT`;

const dockerCompose = `# docker-compose.yml - Local Development Stack

version: '3.8'

services:
  # ================================
  # Message Queue
  # ================================
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"

  # ================================
  # Databases
  # ================================
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: cicd
      POSTGRES_PASSWORD: cicd123
      POSTGRES_DB: anomaly_detection
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # ================================
  # Observability
  # ================================
  prometheus:
    image: prom/prometheus:v2.48.0
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
      - loki

  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:2.9.0
    volumes:
      - ./config/promtail.yml:/etc/promtail/config.yml
      - /var/log:/var/log
    command: -config.file=/etc/promtail/config.yml

  # ================================
  # ML Scoring Service
  # ================================
  ml-scoring:
    build:
      context: ./ml-anomaly-detector
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: /app/models/v1
    volumes:
      - ./ml-anomaly-detector/models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ================================
  # ETL Pipeline
  # ================================
  etl:
    build:
      context: ./etl-pipeline
      dockerfile: Dockerfile
    environment:
      KAFKA_BROKERS: kafka:9092
      KAFKA_TOPIC: ci-pipeline-events
      S3_BUCKET: local-features
    depends_on:
      - kafka
    # For local dev, use MinIO instead of S3
    # volumes:
    #   - ./data:/app/data

  # ================================
  # MinIO (S3 compatible storage)
  # ================================
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
  minio_data:`;

const minikubeCommands = `#!/bin/bash
# minikube-setup.sh - Setup Kubernetes locally

set -e

echo "=== Setting up Minikube for CI/CD Anomaly Detection ==="

# Start minikube with sufficient resources
minikube start \\
  --memory=8192 \\
  --cpus=4 \\
  --disk-size=50g \\
  --kubernetes-version=v1.28.0

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Create namespaces
kubectl create namespace ml
kubectl create namespace monitoring
kubectl create namespace argocd

# Install ArgoCD
echo "Installing ArgoCD..."
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s

# Get ArgoCD password
echo "ArgoCD initial password:"
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""

# Install Prometheus + Grafana
echo "Installing Prometheus stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack \\
  -n monitoring \\
  --set grafana.adminPassword=admin

# Install Loki
echo "Installing Loki..."
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack -n monitoring

# Build and load ML image
echo "Building ML service image..."
eval \$(minikube docker-env)
docker build -t ml-scoring:latest ./ml-anomaly-detector

# Apply ML service manifests
echo "Deploying ML service..."
kubectl apply -f k8s/ml-scoring/

# Wait for ML service
kubectl wait --for=condition=available deployment/ml-scoring -n ml --timeout=120s

# Get service URLs
echo ""
echo "=== Service URLs ==="
echo "ML Scoring: \$(minikube service ml-scoring -n ml --url)"
echo "Grafana: \$(minikube service prometheus-grafana -n monitoring --url)"
echo "ArgoCD: kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo ""
echo "Setup complete! 🚀"`;

export function ReadmeSection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Quick Start</h1>
        <p className="text-muted-foreground">
          Commands to run the entire MVP locally with Docker Compose or Minikube
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="section-card text-center">
          <Terminal className="w-8 h-8 text-primary mx-auto mb-2" />
          <div className="font-semibold">Docker Compose</div>
          <div className="text-xs text-muted-foreground">Local dev</div>
        </div>
        <div className="section-card text-center">
          <Settings className="w-8 h-8 text-accent mx-auto mb-2" />
          <div className="font-semibold">Minikube</div>
          <div className="text-xs text-muted-foreground">K8s local</div>
        </div>
        <div className="section-card text-center">
          <Rocket className="w-8 h-8 text-success mx-auto mb-2" />
          <div className="font-semibold">ML API</div>
          <div className="text-xs text-muted-foreground">:8000/score</div>
        </div>
        <div className="section-card text-center">
          <BookOpen className="w-8 h-8 text-warning mx-auto mb-2" />
          <div className="font-semibold">Grafana</div>
          <div className="text-xs text-muted-foreground">:3000</div>
        </div>
      </div>

      <SectionCard 
        title="README.md" 
        subtitle="Complete project documentation"
        icon={<BookOpen className="w-5 h-5" />}
        badge="Markdown"
      >
        <CodeBlock code={readme} language="markdown" filename="README.md" />
      </SectionCard>

      <SectionCard 
        title="Docker Compose" 
        subtitle="Local development infrastructure"
        icon={<Terminal className="w-5 h-5" />}
        badge="YAML"
      >
        <CodeBlock code={dockerCompose} language="yaml" filename="docker-compose.yml" />
      </SectionCard>

      <SectionCard 
        title="Minikube Setup" 
        subtitle="Kubernetes local cluster setup"
        icon={<Settings className="w-5 h-5" />}
        badge="bash"
      >
        <CodeBlock code={minikubeCommands} language="bash" filename="minikube-setup.sh" />
      </SectionCard>
    </div>
  );
}

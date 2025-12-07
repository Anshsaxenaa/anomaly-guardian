import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { Activity, BarChart3, FileText, Cpu } from "lucide-react";

const prometheusScrapeConfig = `# prometheus.yml - Scrape Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # GitHub Actions Runner Metrics (self-hosted)
  - job_name: 'github-actions-runner'
    static_configs:
      - targets: ['runner-exporter:9252']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '(.*):(\\d+)'
        replacement: '\${1}'

  # CI Build Metrics (custom exporter)
  - job_name: 'ci-builds'
    metrics_path: /metrics
    static_configs:
      - targets: ['ci-exporter:9090']
    metric_relabel_configs:
      - source_labels: [workflow_name]
        regex: 'build-(.*)'
        target_label: build_type
        replacement: '\${1}'

  # Kubernetes Pod Metrics
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

  # Node Exporter for resource metrics
  - job_name: 'node-exporter'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

  # Application Runtime (NestJS + FastAPI)
  - job_name: 'app-runtime'
    static_configs:
      - targets: 
          - 'nestjs-backend:3000'
          - 'ml-scoring:8000'
    metrics_path: /metrics

  # ArgoCD Metrics
  - job_name: 'argocd'
    static_configs:
      - targets: ['argocd-metrics:8082']`;

const metricsToCollect = `# Key Metrics to Collect

## CI Runner Metrics
- github_runner_job_duration_seconds          # Build time histogram
- github_runner_job_status                    # Success/failure counter
- github_runner_queue_time_seconds            # Time waiting for runner
- github_runner_step_duration_seconds         # Per-step timing

## Build Metrics (Custom)
- ci_build_duration_seconds{workflow,branch,commit}
- ci_build_status{workflow,status="success|failure|cancelled"}
- ci_test_count{workflow,status="passed|failed|skipped"}
- ci_test_duration_seconds{workflow,suite}
- ci_artifact_size_bytes{workflow,artifact_type}
- ci_docker_build_duration_seconds{image,stage}
- ci_docker_layer_cache_hits_total

## Deployment Metrics
- argocd_app_sync_status{app,project}
- argocd_app_health_status{app}
- deployment_rollout_duration_seconds{app,environment}
- deployment_replica_count{app,status="available|unavailable"}

## Resource Metrics (from node-exporter)
- node_cpu_seconds_total
- node_memory_MemAvailable_bytes
- node_disk_io_time_seconds_total
- container_cpu_usage_seconds_total
- container_memory_working_set_bytes

## Application Runtime
- http_request_duration_seconds{method,path,status}
- http_requests_total{method,path,status}
- nodejs_eventloop_lag_seconds
- python_gc_collections_total
- db_query_duration_seconds{query_type}`;

const grafanaDashboardJson = `{
  "dashboard": {
    "title": "CI/CD Anomaly Detection Dashboard",
    "uid": "cicd-anomaly-main",
    "panels": [
      {
        "title": "Build Duration Trend",
        "type": "timeseries",
        "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(ci_build_duration_seconds_bucket[5m])) by (le, workflow))",
          "legendFormat": "{{workflow}} p95"
        }, {
          "expr": "histogram_quantile(0.50, sum(rate(ci_build_duration_seconds_bucket[5m])) by (le, workflow))",
          "legendFormat": "{{workflow}} p50"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": { "lineWidth": 2, "fillOpacity": 10 }
          }
        },
        "options": {
          "tooltip": { "mode": "multi" },
          "legend": { "displayMode": "table", "placement": "right" }
        }
      },
      {
        "title": "Test Failure Rate",
        "type": "timeseries",
        "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
        "targets": [{
          "expr": "sum(rate(ci_test_count{status='failed'}[1h])) by (workflow) / sum(rate(ci_test_count[1h])) by (workflow) * 100",
          "legendFormat": "{{workflow}}"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 5, "color": "yellow" },
                { "value": 15, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "title": "Anomaly Score (ML)",
        "type": "gauge",
        "gridPos": { "x": 0, "y": 8, "w": 6, "h": 6 },
        "targets": [{
          "expr": "ml_anomaly_score{pipeline='main'}"
        }],
        "fieldConfig": {
          "defaults": {
            "min": 0, "max": 1,
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 0.5, "color": "yellow" },
                { "value": 0.8, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "title": "Resource Usage During Build",
        "type": "timeseries",
        "gridPos": { "x": 6, "y": 8, "w": 18, "h": 6 },
        "targets": [{
          "expr": "avg(rate(container_cpu_usage_seconds_total{container=~'runner.*'}[5m])) * 100",
          "legendFormat": "CPU %"
        }, {
          "expr": "avg(container_memory_working_set_bytes{container=~'runner.*'}) / 1024 / 1024 / 1024",
          "legendFormat": "Memory GB"
        }]
      },
      {
        "title": "Deployment Frequency",
        "type": "stat",
        "gridPos": { "x": 0, "y": 14, "w": 4, "h": 4 },
        "targets": [{
          "expr": "sum(increase(argocd_app_sync_total{status='Succeeded'}[24h]))"
        }],
        "options": { "colorMode": "value" }
      },
      {
        "title": "Lead Time for Changes",
        "type": "stat",
        "gridPos": { "x": 4, "y": 14, "w": 4, "h": 4 },
        "targets": [{
          "expr": "avg(deployment_rollout_duration_seconds{environment='production'})"
        }],
        "fieldConfig": { "defaults": { "unit": "s" } }
      }
    ]
  }
}`;

const lokiConfig = `# Loki Stack Configuration

# promtail.yml - Log collection
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # GitHub Actions Logs (via webhook)
  - job_name: github-actions
    loki_push_api:
      server:
        http_listen_port: 3500
      labels:
        job: github-actions
    relabel_configs:
      - source_labels: ['workflow']
        target_label: 'workflow_name'
      - source_labels: ['run_id']
        target_label: 'run_id'

  # Kubernetes Pod Logs
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    pipeline_stages:
      - docker: {}
      - match:
          selector: '{app="ci-runner"}'
          stages:
            - regex:
                expression: 'Step (?P<step>\\d+)/(?P<total>\\d+)'
            - labels:
                step:
                total:
      - match:
          selector: '{app="nestjs-backend"}'
          stages:
            - json:
                expressions:
                  level: level
                  message: message
                  trace_id: trace_id

  # Build artifact logs
  - job_name: build-logs
    static_configs:
      - targets: ['localhost']
        labels:
          job: build-logs
          __path__: /var/log/builds/*.log

# LogQL Queries for Analysis
# ===========================
# Failed builds in last hour:
#   {job="github-actions"} |= "FAILED" | count_over_time([1h])
#
# Slow steps (>5min):
#   {job="github-actions"} | json | step_duration > 300
#
# Error patterns:
#   sum by (error_type) (count_over_time({job="kubernetes-pods"} |= "error" [24h]))`;

export function ObservabilitySection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Observability Stack</h1>
        <p className="text-muted-foreground">
          Prometheus + Grafana + Loki instrumentation for CI/CD pipelines
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="section-card">
          <Activity className="w-5 h-5 text-primary mb-2" />
          <div className="text-sm font-medium">Metrics</div>
          <div className="text-xs text-muted-foreground">Prometheus TSDB</div>
        </div>
        <div className="section-card">
          <FileText className="w-5 h-5 text-accent mb-2" />
          <div className="text-sm font-medium">Logs</div>
          <div className="text-xs text-muted-foreground">Loki + Promtail</div>
        </div>
        <div className="section-card">
          <BarChart3 className="w-5 h-5 text-success mb-2" />
          <div className="text-sm font-medium">Dashboards</div>
          <div className="text-xs text-muted-foreground">Grafana Viz</div>
        </div>
        <div className="section-card">
          <Cpu className="w-5 h-5 text-warning mb-2" />
          <div className="text-sm font-medium">Traces</div>
          <div className="text-xs text-muted-foreground">Tempo (optional)</div>
        </div>
      </div>

      <SectionCard 
        title="Prometheus Scrape Configuration" 
        icon={<Activity className="w-5 h-5" />}
        badge="prometheus.yml"
      >
        <CodeBlock code={prometheusScrapeConfig} language="yaml" filename="prometheus.yml" />
      </SectionCard>

      <SectionCard 
        title="Metrics to Collect" 
        subtitle="From CI runners, builds, tests, deployments, and runtime"
        icon={<BarChart3 className="w-5 h-5" />}
      >
        <CodeBlock code={metricsToCollect} language="yaml" filename="metrics-catalog.md" />
      </SectionCard>

      <SectionCard 
        title="Grafana Dashboard" 
        subtitle="Build duration and test failure trends"
        icon={<BarChart3 className="w-5 h-5" />}
        badge="JSON"
      >
        <CodeBlock code={grafanaDashboardJson} language="json" filename="cicd-dashboard.json" />
      </SectionCard>

      <SectionCard 
        title="Loki Log Aggregation" 
        icon={<FileText className="w-5 h-5" />}
        badge="promtail.yml"
      >
        <CodeBlock code={lokiConfig} language="yaml" filename="promtail.yml" />
      </SectionCard>
    </div>
  );
}

import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { GitBranch, Workflow, Shield, AlertTriangle } from "lucide-react";

const githubActionsWorkflow = `# .github/workflows/ci-anomaly-detection.yml
name: CI with Anomaly Detection

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  ML_SCORING_URL: \${{ secrets.ML_SCORING_URL }}
  KAFKA_BROKERS: \${{ secrets.KAFKA_BROKERS }}
  SLACK_WEBHOOK: \${{ secrets.SLACK_WEBHOOK }}
  ANOMALY_THRESHOLD: 0.8

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    outputs:
      build_time: \${{ steps.metrics.outputs.build_time }}
      test_results: \${{ steps.test.outputs.results }}
      anomaly_score: \${{ steps.score.outputs.anomaly_score }}
      should_deploy: \${{ steps.gate.outputs.should_deploy }}
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # For commit churn calculation
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      
      - name: Record Start Time
        id: start
        run: echo "start_time=\$(date +%s)" >> \$GITHUB_OUTPUT
      
      # ================================
      # BUILD PHASE
      # ================================
      - name: Install Dependencies
        run: npm ci
      
      - name: Build Application
        id: build
        run: |
          npm run build
          echo "build_status=success" >> \$GITHUB_OUTPUT
        continue-on-error: true
      
      # ================================
      # TEST PHASE
      # ================================
      - name: Run Tests
        id: test
        run: |
          npm run test -- --json --outputFile=test-results.json || true
          
          # Parse test results
          if [ -f test-results.json ]; then
            PASSED=\$(jq '.numPassedTests' test-results.json)
            FAILED=\$(jq '.numFailedTests' test-results.json)
            SKIPPED=\$(jq '.numPendingTests' test-results.json)
          else
            PASSED=0
            FAILED=1
            SKIPPED=0
          fi
          
          echo "passed=\$PASSED" >> \$GITHUB_OUTPUT
          echo "failed=\$FAILED" >> \$GITHUB_OUTPUT
          echo "skipped=\$SKIPPED" >> \$GITHUB_OUTPUT
          echo "results={\\"passed\\":\$PASSED,\\"failed\\":\$FAILED,\\"skipped\\":\$SKIPPED}" >> \$GITHUB_OUTPUT
      
      # ================================
      # COLLECT METRICS
      # ================================
      - name: Collect Build Metrics
        id: metrics
        run: |
          END_TIME=\$(date +%s)
          BUILD_TIME=\$((END_TIME - \${{ steps.start.outputs.start_time }}))
          
          # Get commit churn
          CHURN=\$(git diff --shortstat HEAD~1 HEAD 2>/dev/null | awk '{print \$1}' || echo "0")
          ADDITIONS=\$(git diff --numstat HEAD~1 HEAD 2>/dev/null | awk '{sum+=\$1} END {print sum}' || echo "0")
          DELETIONS=\$(git diff --numstat HEAD~1 HEAD 2>/dev/null | awk '{sum+=\$2} END {print sum}' || echo "0")
          
          # Get resource usage (from runner)
          CPU=\$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | cut -d'%' -f1 || echo "50")
          MEM=\$(free -m | awk 'NR==2{print \$3}' || echo "2048")
          
          echo "build_time=\$BUILD_TIME" >> \$GITHUB_OUTPUT
          echo "commit_churn=\$CHURN" >> \$GITHUB_OUTPUT
          echo "additions=\$ADDITIONS" >> \$GITHUB_OUTPUT
          echo "deletions=\$DELETIONS" >> \$GITHUB_OUTPUT
          echo "cpu=\$CPU" >> \$GITHUB_OUTPUT
          echo "memory=\$MEM" >> \$GITHUB_OUTPUT
      
      # ================================
      # PUSH TO KAFKA (Optional)
      # ================================
      - name: Push Metrics to Pipeline
        if: env.KAFKA_BROKERS != ''
        run: |
          # Install kafka client
          pip install kafka-python
          
          python3 << 'EOF'
          import json
          import os
          from datetime import datetime
          from kafka import KafkaProducer
          
          producer = KafkaProducer(
              bootstrap_servers=os.environ['KAFKA_BROKERS'].split(','),
              value_serializer=lambda v: json.dumps(v).encode('utf-8')
          )
          
          event = {
              "event_id": os.environ['GITHUB_RUN_ID'],
              "event_type": "BUILD_END",
              "timestamp": int(datetime.now().timestamp() * 1000),
              "workflow_run_id": os.environ['GITHUB_RUN_ID'],
              "workflow_name": os.environ['GITHUB_WORKFLOW'],
              "branch": os.environ['GITHUB_REF_NAME'],
              "commit_sha": os.environ['GITHUB_SHA'],
              "actor": os.environ['GITHUB_ACTOR'],
              "metrics": {
                  "duration_ms": int("\${{ steps.metrics.outputs.build_time }}") * 1000,
                  "test_passed": int("\${{ steps.test.outputs.passed }}"),
                  "test_failed": int("\${{ steps.test.outputs.failed }}"),
                  "cpu_percent": float("\${{ steps.metrics.outputs.cpu }}"),
                  "memory_mb": float("\${{ steps.metrics.outputs.memory }}")
              },
              "status": "SUCCESS" if "\${{ steps.build.outputs.build_status }}" == "success" else "FAILURE",
              "labels": {
                  "files_changed": "\${{ steps.metrics.outputs.commit_churn }}",
                  "additions": "\${{ steps.metrics.outputs.additions }}",
                  "deletions": "\${{ steps.metrics.outputs.deletions }}"
              }
          }
          
          producer.send('ci-pipeline-events', event)
          producer.flush()
          print(f"Event sent: {event['event_id']}")
          EOF
      
      # ================================
      # ML ANOMALY SCORING
      # ================================
      - name: Score Build for Anomalies
        id: score
        run: |
          # Prepare scoring payload
          PAYLOAD=\$(cat << EOF
          {
            "build_time_seconds": \${{ steps.metrics.outputs.build_time }},
            "test_fail_count": \${{ steps.test.outputs.failed }},
            "test_fail_rate": \$(echo "scale=4; \${{ steps.test.outputs.failed }} / (\${{ steps.test.outputs.passed }} + \${{ steps.test.outputs.failed }} + 1)" | bc),
            "avg_cpu_percent": \${{ steps.metrics.outputs.cpu }},
            "max_memory_mb": \${{ steps.metrics.outputs.memory }},
            "deploy_delta_hours": 24,
            "commit_churn": \${{ steps.metrics.outputs.commit_churn }},
            "workflow_run_id": "\${{ github.run_id }}",
            "branch": "\${{ github.ref_name }}"
          }
          EOF
          )
          
          echo "Scoring payload: \$PAYLOAD"
          
          # Call ML endpoint
          RESPONSE=\$(curl -s -X POST "\${{ env.ML_SCORING_URL }}/score" \\
            -H "Content-Type: application/json" \\
            -d "\$PAYLOAD" || echo '{"anomaly_score": 0, "is_anomaly": false}')
          
          echo "ML Response: \$RESPONSE"
          
          ANOMALY_SCORE=\$(echo \$RESPONSE | jq -r '.anomaly_score')
          IS_ANOMALY=\$(echo \$RESPONSE | jq -r '.is_anomaly')
          FEATURES=\$(echo \$RESPONSE | jq -c '.contributing_features')
          
          echo "anomaly_score=\$ANOMALY_SCORE" >> \$GITHUB_OUTPUT
          echo "is_anomaly=\$IS_ANOMALY" >> \$GITHUB_OUTPUT
          echo "features=\$FEATURES" >> \$GITHUB_OUTPUT
      
      # ================================
      # DEPLOYMENT GATE
      # ================================
      - name: Deployment Gate
        id: gate
        run: |
          SCORE=\${{ steps.score.outputs.anomaly_score }}
          THRESHOLD=\${{ env.ANOMALY_THRESHOLD }}
          
          if (( \$(echo "\$SCORE > \$THRESHOLD" | bc -l) )); then
            echo "::error::Anomaly detected! Score: \$SCORE (threshold: \$THRESHOLD)"
            echo "should_deploy=false" >> \$GITHUB_OUTPUT
          else
            echo "Build passed anomaly check. Score: \$SCORE"
            echo "should_deploy=true" >> \$GITHUB_OUTPUT
          fi
      
      # ================================
      # ALERT ON ANOMALY
      # ================================
      - name: Send Slack Alert
        if: steps.gate.outputs.should_deploy == 'false'
        run: |
          curl -X POST \${{ env.SLACK_WEBHOOK }} \\
            -H 'Content-Type: application/json' \\
            -d @- << EOF
          {
            "blocks": [
              {
                "type": "header",
                "text": {
                  "type": "plain_text",
                  "text": "🚨 CI/CD Anomaly Detected - Deployment Blocked",
                  "emoji": true
                }
              },
              {
                "type": "section",
                "fields": [
                  {"type": "mrkdwn", "text": "*Workflow:*\\n\${{ github.workflow }}"},
                  {"type": "mrkdwn", "text": "*Branch:*\\n\${{ github.ref_name }}"},
                  {"type": "mrkdwn", "text": "*Anomaly Score:*\\n\${{ steps.score.outputs.anomaly_score }}"},
                  {"type": "mrkdwn", "text": "*Threshold:*\\n\${{ env.ANOMALY_THRESHOLD }}"}
                ]
              },
              {
                "type": "section",
                "text": {
                  "type": "mrkdwn",
                  "text": "*Top Contributing Factors:*\\n\${{ steps.score.outputs.features }}"
                }
              },
              {
                "type": "section",
                "fields": [
                  {"type": "mrkdwn", "text": "*Build Time:*\\n\${{ steps.metrics.outputs.build_time }}s"},
                  {"type": "mrkdwn", "text": "*Test Failures:*\\n\${{ steps.test.outputs.failed }}"},
                  {"type": "mrkdwn", "text": "*Commit Churn:*\\n\${{ steps.metrics.outputs.commit_churn }} files"}
                ]
              },
              {
                "type": "context",
                "elements": [
                  {"type": "mrkdwn", "text": "Run: <https://github.com/\${{ github.repository }}/actions/runs/\${{ github.run_id }}|#\${{ github.run_id }}>"},
                  {"type": "mrkdwn", "text": "Commit: <https://github.com/\${{ github.repository }}/commit/\${{ github.sha }}|\${{ github.sha }}>"}
                ]
              },
              {
                "type": "actions",
                "elements": [
                  {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Logs"},
                    "url": "https://github.com/\${{ github.repository }}/actions/runs/\${{ github.run_id }}"
                  },
                  {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Create JIRA Ticket"},
                    "style": "primary",
                    "url": "https://your-jira.atlassian.net/secure/CreateIssue.jspa?pid=CICD&issuetype=Bug"
                  }
                ]
              }
            ]
          }
          EOF
      
      - name: Fail on Anomaly
        if: steps.gate.outputs.should_deploy == 'false'
        run: exit 1
  
  # ================================
  # DEPLOY (only if no anomaly)
  # ================================
  deploy:
    needs: build-and-test
    if: needs.build-and-test.outputs.should_deploy == 'true' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
      - name: Deploy to Production
        run: |
          echo "Deploying... (ArgoCD sync or kubectl apply)"
          # kubectl apply -f k8s/
          # argocd app sync my-app
      
      - name: Record Deployment
        run: |
          # Push deployment event to Kafka for tracking
          echo "Deployment recorded"`;

const prometheusMetricsExporter = `# ci-metrics-exporter.py
"""
Custom Prometheus exporter for CI metrics
Run alongside GitHub Actions runner or as webhook receiver
"""

from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Metrics
BUILD_DURATION = Histogram(
    'ci_build_duration_seconds',
    'Build duration in seconds',
    ['workflow', 'branch', 'status'],
    buckets=[30, 60, 120, 300, 600, 1200, 1800, 3600]
)

TEST_COUNT = Counter(
    'ci_test_count',
    'Number of tests by status',
    ['workflow', 'status']
)

ANOMALY_SCORE = Gauge(
    'ml_anomaly_score',
    'Latest ML anomaly score',
    ['pipeline', 'branch']
)

BUILD_STATUS = Counter(
    'ci_build_status',
    'Build status counter',
    ['workflow', 'status']
)

def record_build(workflow, branch, duration, status, test_passed, test_failed, anomaly_score):
    """Record build metrics"""
    BUILD_DURATION.labels(workflow=workflow, branch=branch, status=status).observe(duration)
    BUILD_STATUS.labels(workflow=workflow, status=status).inc()
    TEST_COUNT.labels(workflow=workflow, status='passed').inc(test_passed)
    TEST_COUNT.labels(workflow=workflow, status='failed').inc(test_failed)
    ANOMALY_SCORE.labels(pipeline=workflow, branch=branch).set(anomaly_score)

if __name__ == '__main__':
    start_http_server(9090)
    print("Metrics server running on :9090")
    while True:
        time.sleep(1)`;

export function CIIntegrationSection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">CI Integration</h1>
        <p className="text-muted-foreground">
          GitHub Actions workflow with ML scoring and deployment gates
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="section-card text-center">
          <GitBranch className="w-8 h-8 text-primary mx-auto mb-2" />
          <div className="font-semibold">Build + Test</div>
          <div className="text-xs text-muted-foreground">npm ci, test</div>
        </div>
        <div className="section-card text-center">
          <Workflow className="w-8 h-8 text-accent mx-auto mb-2" />
          <div className="font-semibold">Collect Metrics</div>
          <div className="text-xs text-muted-foreground">Duration, CPU, Mem</div>
        </div>
        <div className="section-card text-center">
          <Shield className="w-8 h-8 text-success mx-auto mb-2" />
          <div className="font-semibold">ML Score</div>
          <div className="text-xs text-muted-foreground">/score endpoint</div>
        </div>
        <div className="section-card text-center">
          <AlertTriangle className="w-8 h-8 text-warning mx-auto mb-2" />
          <div className="font-semibold">Gate Deploy</div>
          <div className="text-xs text-muted-foreground">score {">"} 0.8 = block</div>
        </div>
      </div>

      <SectionCard 
        title="GitHub Actions Workflow" 
        subtitle="Complete CI pipeline with anomaly detection"
        icon={<GitBranch className="w-5 h-5" />}
        badge="YAML"
      >
        <CodeBlock code={githubActionsWorkflow} language="yaml" filename=".github/workflows/ci-anomaly-detection.yml" />
      </SectionCard>

      <SectionCard 
        title="Prometheus Metrics Exporter" 
        subtitle="Custom exporter for CI metrics"
        icon={<Workflow className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={prometheusMetricsExporter} language="python" filename="ci-metrics-exporter.py" />
      </SectionCard>
    </div>
  );
}

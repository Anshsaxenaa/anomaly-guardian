import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { Shield, Lock, Eye, Key } from "lucide-react";

const securityChecklist = `# Security & Privacy Checklist for CI/CD Telemetry and ML

## 1. PII Removal

### Data Collection
- [ ] **Hash usernames/emails** before storage (SHA-256 with salt)
- [ ] **Remove IP addresses** from logs or hash them
- [ ] **Scrub commit messages** for sensitive content (API keys, passwords)
- [ ] **Anonymize actor IDs** in training data
- [ ] **Filter file paths** that may contain PII

### Implementation
\`\`\`python
import hashlib
import os

SALT = os.environ.get('PII_SALT', 'your-secret-salt')

def anonymize(value: str) -> str:
    """Hash PII with salt for anonymization"""
    return hashlib.sha256(f"{SALT}{value}".encode()).hexdigest()[:16]

def scrub_commit_message(message: str) -> str:
    """Remove potential secrets from commit messages"""
    import re
    patterns = [
        r'(?i)(api[_-]?key|secret|password|token)[=:]\s*[\'"]?[\w-]+',
        r'[A-Za-z0-9+/]{40,}={0,2}',  # Base64 strings
        r'sk-[a-zA-Z0-9]{48}',         # OpenAI keys
        r'ghp_[a-zA-Z0-9]{36}',        # GitHub tokens
    ]
    for pattern in patterns:
        message = re.sub(pattern, '[REDACTED]', message)
    return message
\`\`\`

## 2. Encryption

### At Rest
- [ ] **S3 bucket encryption** - SSE-S3 or SSE-KMS
- [ ] **Database encryption** - RDS encryption enabled
- [ ] **Model artifacts** - Encrypted before storage

### In Transit
- [ ] **TLS 1.3** for all API endpoints
- [ ] **mTLS** for internal service communication
- [ ] **Kafka SSL** for event streaming

### Configuration
\`\`\`yaml
# S3 Bucket Policy
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyUnencryptedUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::cicd-anomaly-data/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    }
  ]
}
\`\`\`

## 3. Access Control

### Principle of Least Privilege
- [ ] **ML service** - Read-only access to feature store
- [ ] **ETL pipeline** - Write to features, no model access
- [ ] **CI runners** - Write to raw events only
- [ ] **Dashboards** - Read-only aggregated metrics

### IAM Policies
\`\`\`json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::cicd-anomaly-data/features/*"
      ],
      "Condition": {
        "StringEquals": {
          "aws:PrincipalTag/service": "ml-scoring"
        }
      }
    }
  ]
}
\`\`\`

### Authentication
- [ ] **API keys** for external services (GitHub, Slack)
- [ ] **Service accounts** for internal services (Kubernetes)
- [ ] **OIDC** for human access (ArgoCD, Grafana)
- [ ] **Rotate secrets** every 90 days

## 4. Data Retention

| Data Type | Retention | Storage | Justification |
|-----------|-----------|---------|---------------|
| Raw events | 7 days | S3 Standard | Debugging |
| Features | 90 days | S3 IA | Model training |
| Predictions | 30 days | S3 Standard | Audit trail |
| Models | Indefinite | S3 Versioned | Reproducibility |
| Logs | 30 days | Loki | Troubleshooting |

## 5. Audit Logging

- [ ] **CloudTrail** for AWS API calls
- [ ] **Kubernetes audit logs** for cluster access
- [ ] **Application logs** for model predictions
- [ ] **Alert on anomalous access patterns**

\`\`\`python
# Audit log entry for ML predictions
{
    "timestamp": "2024-01-15T14:32:15Z",
    "service": "ml-scoring",
    "action": "predict",
    "workflow_run_id": "4521",  # Anonymized
    "anomaly_score": 0.87,
    "decision": "blocked",
    "model_version": "v1.2.0",
    "request_id": "req-abc123",
    "client_ip_hash": "a1b2c3..."  # Hashed
}
\`\`\`

## 6. Model Security

### Training Data
- [ ] **Validate input data** for poisoning attacks
- [ ] **Monitor training data distribution** for drift
- [ ] **Secure training environment** (isolated, no internet)

### Inference
- [ ] **Input validation** - Schema enforcement
- [ ] **Rate limiting** - Prevent abuse
- [ ] **Output sanitization** - No raw scores in errors

### Model Artifacts
- [ ] **Sign models** with SHA-256 checksum
- [ ] **Version control** all model changes
- [ ] **Scan for vulnerabilities** in dependencies

## 7. Incident Response

### Detection
- [ ] Alert on unusual API patterns
- [ ] Monitor for model performance degradation
- [ ] Track failed authentication attempts

### Response
- [ ] Documented runbook for security incidents
- [ ] Ability to disable ML scoring without breaking CI
- [ ] Contact list for security team

## 8. Compliance Considerations

### SOC 2 Type II
- [ ] Access controls documented and enforced
- [ ] Change management for model updates
- [ ] Incident response procedures tested

### GDPR (if applicable)
- [ ] Data minimization - only collect needed telemetry
- [ ] Right to erasure - ability to delete user data
- [ ] Data processing agreement with vendors

## 9. Security Testing

- [ ] **SAST** - SonarQube on ML code
- [ ] **Dependency scanning** - Snyk/Trivy on containers
- [ ] **Penetration testing** - Annual assessment
- [ ] **Chaos engineering** - Test failure modes`;

const networkPolicy = `# Kubernetes Network Policies for ML Service

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-scoring-policy
  namespace: ml
spec:
  podSelector:
    matchLabels:
      app: ml-scoring
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Only allow from CI runners and internal services
    - from:
        - namespaceSelector:
            matchLabels:
              name: ci
        - podSelector:
            matchLabels:
              app: nestjs-backend
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # Only allow to S3 (via VPC endpoint) and logging
    - to:
        - ipBlock:
            cidr: 10.0.0.0/8  # VPC CIDR
      ports:
        - protocol: TCP
          port: 443
    # DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-default
  namespace: ml
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress`;

export function SecuritySection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Security & Privacy</h1>
        <p className="text-muted-foreground">
          Comprehensive checklist for telemetry and ML system security
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="section-card text-center">
          <Eye className="w-8 h-8 text-primary mx-auto mb-2" />
          <div className="font-semibold">PII Removal</div>
          <div className="text-xs text-muted-foreground">Anonymization</div>
        </div>
        <div className="section-card text-center">
          <Lock className="w-8 h-8 text-accent mx-auto mb-2" />
          <div className="font-semibold">Encryption</div>
          <div className="text-xs text-muted-foreground">At rest + transit</div>
        </div>
        <div className="section-card text-center">
          <Key className="w-8 h-8 text-success mx-auto mb-2" />
          <div className="font-semibold">Access Control</div>
          <div className="text-xs text-muted-foreground">Least privilege</div>
        </div>
        <div className="section-card text-center">
          <Shield className="w-8 h-8 text-warning mx-auto mb-2" />
          <div className="font-semibold">Audit Logs</div>
          <div className="text-xs text-muted-foreground">CloudTrail</div>
        </div>
      </div>

      <SectionCard 
        title="Security & Privacy Checklist" 
        subtitle="Complete security requirements"
        icon={<Shield className="w-5 h-5" />}
        badge="Checklist"
      >
        <CodeBlock code={securityChecklist} language="markdown" filename="security-checklist.md" />
      </SectionCard>

      <SectionCard 
        title="Network Policies" 
        subtitle="Kubernetes network isolation for ML service"
        icon={<Lock className="w-5 h-5" />}
        badge="YAML"
      >
        <CodeBlock code={networkPolicy} language="yaml" filename="network-policy.yaml" />
      </SectionCard>
    </div>
  );
}

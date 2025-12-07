import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { Bell, FileText, MessageSquare } from "lucide-react";

const slackPayload = `// Slack Alert Payload with Context and Next Steps
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
      "type": "divider"
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Summary:* The latest build on \`main\` branch triggered our ML anomaly detection system. The build exhibited unusual patterns that warrant investigation before deployment."
      }
    },
    {
      "type": "section",
      "fields": [
        {
          "type": "mrkdwn",
          "text": "*Anomaly Score:*\\n\`0.87\` (threshold: 0.80)"
        },
        {
          "type": "mrkdwn",
          "text": "*Workflow:*\\nci-main-build"
        },
        {
          "type": "mrkdwn",
          "text": "*Branch:*\\nmain"
        },
        {
          "type": "mrkdwn",
          "text": "*Commit:*\\n\`abc123f\`"
        },
        {
          "type": "mrkdwn",
          "text": "*Author:*\\n@john.doe"
        },
        {
          "type": "mrkdwn",
          "text": "*Run ID:*\\n#4521"
        }
      ]
    },
    {
      "type": "divider"
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*🔍 Top Contributing Factors (from ML model):*"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "1. **build_time_seconds**: \`542s\` (z-score: +2.3) — Build took 80% longer than average\\n2. **test_fail_count**: \`12\` (z-score: +1.8) — 4x normal failure rate\\n3. **commit_churn**: \`47 files\` (z-score: +1.5) — Large change set\\n4. **avg_cpu_percent**: \`94%\` — Near resource limits\\n5. **deploy_delta_hours**: \`0.5h\` — Very rapid deployment cadence"
      }
    },
    {
      "type": "divider"
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*📋 Suggested Next Steps:*\\n\\n1. **Review test failures** — Check \`test-results.json\` for specific failures\\n2. **Check resource usage** — Grafana dashboard: <https://grafana.internal/d/cicd|CI/CD Metrics>\\n3. **Review commit diff** — Large changes may need splitting: <https://github.com/org/repo/commit/abc123f|View Diff>\\n4. **Compare with baseline** — Last 10 successful builds averaged 300s\\n5. **Consider rollback** — If changes are risky, revert and re-deploy incrementally"
      }
    },
    {
      "type": "divider"
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "⏰ Detected at: 2024-01-15 14:32:15 UTC"
        },
        {
          "type": "mrkdwn",
          "text": "📊 Model: IsolationForest v1.2.0"
        }
      ]
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "📋 View Build Logs",
            "emoji": true
          },
          "style": "primary",
          "url": "https://github.com/org/repo/actions/runs/4521"
        },
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "📈 Grafana Dashboard",
            "emoji": true
          },
          "url": "https://grafana.internal/d/cicd"
        },
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "🎫 Create JIRA Ticket",
            "emoji": true
          },
          "style": "danger",
          "url": "https://jira.internal/secure/CreateIssue.jspa?pid=CICD"
        },
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "🔄 Force Deploy",
            "emoji": true
          },
          "url": "https://argocd.internal/applications/app?operation=sync"
        }
      ]
    }
  ]
}`;

const jiraTemplate = `{
  "fields": {
    "project": {
      "key": "CICD"
    },
    "summary": "[Anomaly] Build #4521 blocked - Score 0.87 on main branch",
    "description": {
      "type": "doc",
      "version": 1,
      "content": [
        {
          "type": "heading",
          "attrs": { "level": 2 },
          "content": [{ "type": "text", "text": "Anomaly Detection Alert" }]
        },
        {
          "type": "paragraph",
          "content": [
            { "type": "text", "text": "Our ML-based CI/CD anomaly detection system blocked deployment for build " },
            { "type": "text", "text": "#4521", "marks": [{ "type": "strong" }] },
            { "type": "text", "text": " due to unusual patterns." }
          ]
        },
        {
          "type": "heading",
          "attrs": { "level": 3 },
          "content": [{ "type": "text", "text": "Key Metrics" }]
        },
        {
          "type": "table",
          "attrs": { "isNumberColumnEnabled": false, "layout": "default" },
          "content": [
            {
              "type": "tableRow",
              "content": [
                { "type": "tableHeader", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Metric" }] }] },
                { "type": "tableHeader", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Value" }] }] },
                { "type": "tableHeader", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Baseline" }] }] },
                { "type": "tableHeader", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Z-Score" }] }] }
              ]
            },
            {
              "type": "tableRow",
              "content": [
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Build Time" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "542s" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "300s avg" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "+2.3", "marks": [{ "type": "textColor", "attrs": { "color": "#de350b" } }] }] }] }
              ]
            },
            {
              "type": "tableRow",
              "content": [
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Test Failures" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "12" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "3 avg" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "+1.8", "marks": [{ "type": "textColor", "attrs": { "color": "#de350b" } }] }] }] }
              ]
            },
            {
              "type": "tableRow",
              "content": [
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Commit Churn" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "47 files" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "8 files avg" }] }] },
                { "type": "tableCell", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "+1.5", "marks": [{ "type": "textColor", "attrs": { "color": "#ff991f" } }] }] }] }
              ]
            }
          ]
        },
        {
          "type": "heading",
          "attrs": { "level": 3 },
          "content": [{ "type": "text", "text": "Root Cause Analysis (RCA)" }]
        },
        {
          "type": "bulletList",
          "content": [
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Large PR merged with 47 file changes" }] }]
            },
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "New dependency added causing longer install times" }] }]
            },
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Test suite includes new integration tests hitting slow endpoints" }] }]
            }
          ]
        },
        {
          "type": "heading",
          "attrs": { "level": 3 },
          "content": [{ "type": "text", "text": "Recommended Actions" }]
        },
        {
          "type": "orderedList",
          "content": [
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Review failing tests in test-results.json artifact" }] }]
            },
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Consider splitting large PR into smaller changes" }] }]
            },
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Add caching for new dependencies" }] }]
            },
            {
              "type": "listItem",
              "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Parallelize integration tests" }] }]
            }
          ]
        },
        {
          "type": "heading",
          "attrs": { "level": 3 },
          "content": [{ "type": "text", "text": "Links" }]
        },
        {
          "type": "bulletList",
          "content": [
            { "type": "listItem", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Build logs: ", "marks": [{ "type": "link", "attrs": { "href": "https://github.com/org/repo/actions/runs/4521" } }] }] }] },
            { "type": "listItem", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Commit: ", "marks": [{ "type": "link", "attrs": { "href": "https://github.com/org/repo/commit/abc123f" } }] }] }] },
            { "type": "listItem", "content": [{ "type": "paragraph", "content": [{ "type": "text", "text": "Grafana dashboard: ", "marks": [{ "type": "link", "attrs": { "href": "https://grafana.internal/d/cicd" } }] }] }] }
          ]
        }
      ]
    },
    "issuetype": {
      "name": "Bug"
    },
    "priority": {
      "name": "High"
    },
    "labels": ["cicd", "anomaly-detection", "ml", "automated"],
    "components": [{ "name": "CI/CD Pipeline" }],
    "assignee": {
      "accountId": "on-call-engineer-id"
    },
    "customfield_10001": "2024-01-16",  // Due date
    "customfield_10002": {              // Severity
      "value": "P1 - Critical"
    }
  }
}`;

const rcaChecklist = `# Root Cause Analysis (RCA) Checklist

## Immediate Actions (within 15 minutes)

- [ ] **Acknowledge alert** in Slack thread
- [ ] **Check build logs** for obvious errors
- [ ] **Review test failures** - are they flaky or real?
- [ ] **Check resource graphs** - CPU/memory spikes?
- [ ] **Identify commit** - what changed?

## Investigation (within 1 hour)

### 1. Build Duration Analysis
- [ ] Compare with last 10 successful builds
- [ ] Check npm install times (dependency changes?)
- [ ] Check Docker build times (cache miss?)
- [ ] Review parallel step execution

### 2. Test Failure Analysis  
- [ ] Identify failing tests
- [ ] Check if failures are flaky (re-run history)
- [ ] Review test logs for root cause
- [ ] Check external dependencies (APIs, databases)

### 3. Resource Analysis
- [ ] Review Grafana CPU/memory graphs
- [ ] Check for resource contention
- [ ] Review runner specifications
- [ ] Check concurrent workflows

### 4. Code Change Analysis
- [ ] Review commit diff size
- [ ] Check for risky patterns:
  - Large refactors
  - New dependencies
  - Database migrations
  - API changes
- [ ] Review PR comments/reviews

## Resolution

### If False Positive
- [ ] Retrain model with this sample
- [ ] Adjust contamination parameter
- [ ] Consider feature engineering improvements
- [ ] Document in runbook

### If True Positive
- [ ] Fix root cause (tests, performance, etc.)
- [ ] Consider reverting if blocking
- [ ] Add regression tests
- [ ] Update monitoring/alerts

## Post-Incident

- [ ] Update JIRA ticket with findings
- [ ] Share summary in #engineering
- [ ] Add to weekly review agenda
- [ ] Update runbook if needed`;

export function AlertsSection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Alerts & RCA</h1>
        <p className="text-muted-foreground">
          Slack notifications, JIRA templates, and root cause analysis workflows
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="section-card text-center">
          <MessageSquare className="w-8 h-8 text-primary mx-auto mb-2" />
          <div className="font-semibold">Slack Alerts</div>
          <div className="text-xs text-muted-foreground">Rich Block Kit</div>
        </div>
        <div className="section-card text-center">
          <FileText className="w-8 h-8 text-accent mx-auto mb-2" />
          <div className="font-semibold">JIRA Tickets</div>
          <div className="text-xs text-muted-foreground">ADF Format</div>
        </div>
        <div className="section-card text-center">
          <Bell className="w-8 h-8 text-warning mx-auto mb-2" />
          <div className="font-semibold">RCA Checklist</div>
          <div className="text-xs text-muted-foreground">Runbook</div>
        </div>
      </div>

      <SectionCard 
        title="Slack Alert Payload" 
        subtitle="Block Kit message with context and actions"
        icon={<MessageSquare className="w-5 h-5" />}
        badge="JSON"
      >
        <CodeBlock code={slackPayload} language="json" filename="slack-alert.json" />
      </SectionCard>

      <SectionCard 
        title="JIRA Ticket Template" 
        subtitle="Atlassian Document Format (ADF)"
        icon={<FileText className="w-5 h-5" />}
        badge="JSON"
      >
        <CodeBlock code={jiraTemplate} language="json" filename="jira-ticket.json" />
      </SectionCard>

      <SectionCard 
        title="RCA Checklist" 
        subtitle="Root cause analysis runbook"
        icon={<Bell className="w-5 h-5" />}
      >
        <CodeBlock code={rcaChecklist} language="markdown" filename="rca-checklist.md" />
      </SectionCard>
    </div>
  );
}

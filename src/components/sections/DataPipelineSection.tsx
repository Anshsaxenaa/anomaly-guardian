import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { Database, Workflow, HardDrive } from "lucide-react";

const kafkaSchema = `// Kafka Topic: ci-pipeline-events
// Avro Schema for Pipeline Events

{
  "type": "record",
  "name": "PipelineEvent",
  "namespace": "com.ecommerce.cicd",
  "fields": [
    {
      "name": "event_id",
      "type": "string",
      "doc": "UUID for this event"
    },
    {
      "name": "event_type",
      "type": {
        "type": "enum",
        "name": "EventType",
        "symbols": ["BUILD_START", "BUILD_END", "TEST_START", "TEST_END", 
                    "DEPLOY_START", "DEPLOY_END", "STEP_COMPLETE"]
      }
    },
    {
      "name": "timestamp",
      "type": "long",
      "logicalType": "timestamp-millis"
    },
    {
      "name": "workflow_run_id",
      "type": "string"
    },
    {
      "name": "workflow_name",
      "type": "string"
    },
    {
      "name": "branch",
      "type": "string"
    },
    {
      "name": "commit_sha",
      "type": "string"
    },
    {
      "name": "actor",
      "type": "string",
      "doc": "GitHub username who triggered"
    },
    {
      "name": "metrics",
      "type": {
        "type": "record",
        "name": "Metrics",
        "fields": [
          {"name": "duration_ms", "type": ["null", "long"], "default": null},
          {"name": "cpu_percent", "type": ["null", "float"], "default": null},
          {"name": "memory_mb", "type": ["null", "float"], "default": null},
          {"name": "test_passed", "type": ["null", "int"], "default": null},
          {"name": "test_failed", "type": ["null", "int"], "default": null},
          {"name": "test_skipped", "type": ["null", "int"], "default": null},
          {"name": "artifact_size_bytes", "type": ["null", "long"], "default": null}
        ]
      }
    },
    {
      "name": "status",
      "type": {
        "type": "enum",
        "name": "Status",
        "symbols": ["RUNNING", "SUCCESS", "FAILURE", "CANCELLED", "TIMEOUT"]
      }
    },
    {
      "name": "error_message",
      "type": ["null", "string"],
      "default": null
    },
    {
      "name": "labels",
      "type": {
        "type": "map",
        "values": "string"
      },
      "default": {}
    }
  ]
}`;

const pythonEtl = `#!/usr/bin/env python3
"""
ETL Pipeline: Convert CI/CD events to ML-ready features
Reads from Kafka, processes, writes to S3 as Parquet
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kafka import KafkaConsumer
import boto3
from dataclasses import dataclass, asdict


@dataclass
class BuildFeatures:
    """Feature vector for ML model"""
    workflow_run_id: str
    timestamp: datetime
    
    # Duration features
    build_time_seconds: float
    build_time_zscore: float  # Normalized against historical
    
    # Test features
    test_fail_count: int
    test_fail_rate: float
    test_duration_seconds: float
    
    # Resource features
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_mb: float
    max_memory_mb: float
    
    # Deployment features
    deploy_delta_seconds: float  # Time since last deploy
    deploy_frequency_24h: int    # Deploys in last 24h
    
    # Code churn features
    commit_churn: int           # Files changed
    commit_additions: int
    commit_deletions: int
    
    # Contextual
    is_weekend: bool
    hour_of_day: int
    branch: str
    actor_hash: str  # Anonymized actor


class FeatureStore:
    """Simple feature store for historical data"""
    
    def __init__(self, s3_bucket: str, prefix: str = "features"):
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        self.prefix = prefix
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_historical_stats(self, workflow: str, days: int = 30) -> Dict:
        """Get historical stats for z-score calculation"""
        key = f"{self.prefix}/stats/{workflow}_stats.json"
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj['Body'].read())
        except self.s3.exceptions.NoSuchKey:
            return {"mean_build_time": 300, "std_build_time": 60}
    
    def save_features(self, features: List[BuildFeatures], partition_date: str):
        """Save features as partitioned Parquet"""
        df = pd.DataFrame([asdict(f) for f in features])
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # Write to S3 with partitioning
        key = f"{self.prefix}/build_features/date={partition_date}/data.parquet"
        
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer, compression='snappy')
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue().to_pybytes()
        )
        
        print(f"Saved {len(features)} features to s3://{self.bucket}/{key}")


class PipelineETL:
    """Main ETL processor"""
    
    def __init__(
        self,
        kafka_brokers: str,
        kafka_topic: str,
        s3_bucket: str,
        consumer_group: str = "cicd-etl"
    ):
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=kafka_brokers.split(','),
            group_id=consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        self.feature_store = FeatureStore(s3_bucket)
        self.event_buffer: Dict[str, List[dict]] = {}
    
    def _anonymize(self, value: str) -> str:
        """Hash PII for privacy"""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _calculate_zscore(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score for anomaly detection"""
        if std == 0:
            return 0.0
        return (value - mean) / std
    
    def process_event(self, event: dict) -> Optional[BuildFeatures]:
        """Process a single event, return features if build complete"""
        run_id = event['workflow_run_id']
        event_type = event['event_type']
        
        # Buffer events by run_id
        if run_id not in self.event_buffer:
            self.event_buffer[run_id] = []
        self.event_buffer[run_id].append(event)
        
        # Only generate features on BUILD_END
        if event_type != 'BUILD_END':
            return None
        
        events = self.event_buffer.pop(run_id, [])
        if not events:
            return None
        
        # Aggregate metrics from all events
        start_event = next((e for e in events if e['event_type'] == 'BUILD_START'), None)
        end_event = event
        test_events = [e for e in events if e['event_type'] in ('TEST_START', 'TEST_END')]
        
        if not start_event:
            return None
        
        # Calculate duration
        start_ts = datetime.fromtimestamp(start_event['timestamp'] / 1000)
        end_ts = datetime.fromtimestamp(end_event['timestamp'] / 1000)
        build_time = (end_ts - start_ts).total_seconds()
        
        # Get historical stats for normalization
        stats = self.feature_store.get_historical_stats(event['workflow_name'])
        
        # Aggregate test results
        test_failed = sum(e['metrics'].get('test_failed', 0) or 0 for e in events)
        test_passed = sum(e['metrics'].get('test_passed', 0) or 0 for e in events)
        test_total = test_failed + test_passed
        
        # Aggregate resources
        cpu_values = [e['metrics'].get('cpu_percent', 0) or 0 for e in events]
        mem_values = [e['metrics'].get('memory_mb', 0) or 0 for e in events]
        
        return BuildFeatures(
            workflow_run_id=run_id,
            timestamp=end_ts,
            build_time_seconds=build_time,
            build_time_zscore=self._calculate_zscore(
                build_time,
                stats['mean_build_time'],
                stats['std_build_time']
            ),
            test_fail_count=test_failed,
            test_fail_rate=test_failed / test_total if test_total > 0 else 0.0,
            test_duration_seconds=sum(
                (e['metrics'].get('duration_ms', 0) or 0) / 1000
                for e in test_events
            ),
            avg_cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            max_cpu_percent=max(cpu_values) if cpu_values else 0,
            avg_memory_mb=sum(mem_values) / len(mem_values) if mem_values else 0,
            max_memory_mb=max(mem_values) if mem_values else 0,
            deploy_delta_seconds=0.0,  # TODO: Calculate from deploy events
            deploy_frequency_24h=0,    # TODO: Query historical
            commit_churn=event.get('labels', {}).get('files_changed', 0),
            commit_additions=event.get('labels', {}).get('additions', 0),
            commit_deletions=event.get('labels', {}).get('deletions', 0),
            is_weekend=end_ts.weekday() >= 5,
            hour_of_day=end_ts.hour,
            branch=event['branch'],
            actor_hash=self._anonymize(event['actor'])
        )
    
    def run(self, batch_size: int = 100, flush_interval_seconds: int = 60):
        """Main processing loop"""
        features_batch: List[BuildFeatures] = []
        last_flush = datetime.now()
        
        print(f"Starting ETL consumer...")
        
        for message in self.consumer:
            event = message.value
            
            try:
                features = self.process_event(event)
                if features:
                    features_batch.append(features)
            except Exception as e:
                print(f"Error processing event: {e}")
                continue
            
            # Flush conditions
            should_flush = (
                len(features_batch) >= batch_size or
                (datetime.now() - last_flush).seconds >= flush_interval_seconds
            )
            
            if should_flush and features_batch:
                partition_date = datetime.now().strftime('%Y-%m-%d')
                self.feature_store.save_features(features_batch, partition_date)
                features_batch = []
                last_flush = datetime.now()


if __name__ == "__main__":
    etl = PipelineETL(
        kafka_brokers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
        kafka_topic=os.getenv("KAFKA_TOPIC", "ci-pipeline-events"),
        s3_bucket=os.getenv("S3_BUCKET", "cicd-anomaly-data")
    )
    etl.run()`;

const storagePlan = `# Data Storage Plan

## S3 Bucket Structure
\`\`\`
s3://cicd-anomaly-data/
├── raw/                          # Raw events (7-day retention)
│   └── ci-events/
│       └── date=YYYY-MM-DD/
│           └── hour=HH/
│               └── events_*.json.gz
│
├── features/                     # Processed features (90-day retention)
│   ├── build_features/
│   │   └── date=YYYY-MM-DD/
│   │       └── data.parquet
│   └── stats/
│       └── {workflow}_stats.json
│
├── models/                       # Trained models (versioned)
│   └── isolation_forest/
│       └── v{version}/
│           ├── model.joblib
│           ├── metadata.json
│           └── metrics.json
│
└── predictions/                  # Inference results (30-day retention)
    └── date=YYYY-MM-DD/
        └── scores.parquet
\`\`\`

## Parquet Schema (build_features)
| Column              | Type     | Description                    |
|---------------------|----------|--------------------------------|
| workflow_run_id     | string   | Unique identifier              |
| timestamp           | timestamp| Event time                     |
| build_time_seconds  | float    | Total build duration           |
| build_time_zscore   | float    | Normalized build time          |
| test_fail_count     | int      | Number of failed tests         |
| test_fail_rate      | float    | Failure ratio                  |
| avg_cpu_percent     | float    | Mean CPU usage                 |
| max_memory_mb       | float    | Peak memory usage              |
| commit_churn        | int      | Files changed in commit        |
| is_weekend          | bool     | Weekend flag                   |
| hour_of_day         | int      | 0-23                           |
| branch              | string   | Git branch                     |
| actor_hash          | string   | Anonymized user                |

## Lifecycle Policies
- Raw events: 7 days → Glacier → Delete after 30 days
- Features: 90 days in Standard → Glacier Deep Archive
- Models: Kept indefinitely with versioning
- Predictions: 30 days → Delete

## Alternative: Lightweight (No Kafka)
For smaller scale, replace Kafka with:
1. **Redis Streams** - Built-in, simpler, 10K events/sec
2. **AWS SQS + Lambda** - Serverless, auto-scaling
3. **Direct webhook → S3** - Simplest, batch process hourly`;

export function DataPipelineSection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">Data Pipeline</h1>
        <p className="text-muted-foreground">
          Event ingestion, ETL processing, and feature storage
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="section-card text-center">
          <Database className="w-8 h-8 text-primary mx-auto mb-2" />
          <div className="font-semibold">Kafka</div>
          <div className="text-xs text-muted-foreground">Event Streaming</div>
        </div>
        <div className="section-card text-center">
          <Workflow className="w-8 h-8 text-accent mx-auto mb-2" />
          <div className="font-semibold">Python ETL</div>
          <div className="text-xs text-muted-foreground">Feature Engineering</div>
        </div>
        <div className="section-card text-center">
          <HardDrive className="w-8 h-8 text-success mx-auto mb-2" />
          <div className="font-semibold">S3 + Parquet</div>
          <div className="text-xs text-muted-foreground">Data Lake</div>
        </div>
      </div>

      <SectionCard 
        title="Kafka Event Schema" 
        subtitle="Avro schema for CI pipeline events"
        icon={<Database className="w-5 h-5" />}
        badge="Avro"
      >
        <CodeBlock code={kafkaSchema} language="json" filename="pipeline_event.avsc" />
      </SectionCard>

      <SectionCard 
        title="Python ETL Pipeline" 
        subtitle="Kafka consumer → Feature extraction → S3 Parquet"
        icon={<Workflow className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={pythonEtl} language="python" filename="etl_pipeline.py" />
      </SectionCard>

      <SectionCard 
        title="Storage Plan" 
        subtitle="S3 bucket structure and lifecycle policies"
        icon={<HardDrive className="w-5 h-5" />}
      >
        <CodeBlock code={storagePlan} language="markdown" filename="storage-plan.md" />
      </SectionCard>
    </div>
  );
}

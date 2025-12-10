# GCP Dataproc Deployment Guide

## Overview

This guide covers running the EmoArc pipeline on GCP Dataproc to process all **~75,000 English books** from Project Gutenberg.

---

## Cost & Time Estimates

**Cluster:** 1 master + 4 workers (n2-highmem-8)
- **Hourly cost:** $2.78/hour
- **Estimated runtime:** 3-8 hours
- **Total cost:** $8-22 (one-time processing)

**Storage:** GCS Standard
- **Input data:** ~50 GB (books + lexicons)
- **Output data:** ~2 GB (trajectories CSV)
- **Storage cost:** $0.02/GB/month = ~$1/month

---

## Prerequisites

1. **GCP Project** with billing enabled
2. **Dataproc API** enabled: `gcloud services enable dataproc.googleapis.com`
3. **GCS bucket** for data storage
4. **gcloud CLI** installed and authenticated

---

## Step 1: Prepare Data

### Upload to Google Cloud Storage

```bash
# Create bucket (choose your region)
gsutil mb -l us-central1 gs://YOUR-BUCKET-NAME

# Upload books directory (this may take a while)
gsutil -m cp -r data/books gs://YOUR-BUCKET-NAME/data/
gsutil cp data/gutenberg_metadata.csv gs://YOUR-BUCKET-NAME/data/
gsutil cp data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt gs://YOUR-BUCKET-NAME/data/
gsutil cp data/NRC-VAD-Lexicon-v2.1.txt gs://YOUR-BUCKET-NAME/data/

# Upload pipeline code
tar -czf emoarc-pipeline.tar.gz src/ main.py
gsutil cp emoarc-pipeline.tar.gz gs://YOUR-BUCKET-NAME/code/
gsutil cp main.py gs://YOUR-BUCKET-NAME/code/

# Verify uploads
gsutil du -sh gs://YOUR-BUCKET-NAME/*
```

---

## Step 2: Create Dataproc Cluster

### Option A: Balanced Cluster (Recommended)

Good for most use cases, balances cost and performance.

```bash
gcloud dataproc clusters create emoarc-cluster \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n2-highmem-8 \
  --master-boot-disk-size=100GB \
  --num-workers=4 \
  --worker-machine-type=n2-highmem-8 \
  --worker-boot-disk-size=100GB \
  --image-version=2.2-debian12 \
  --max-idle=30m \
  --enable-component-gateway \
  --properties=^#^spark:spark.driver.memory=24g#spark:spark.executor.memory=24g#spark:spark.executor.cores=4#spark:spark.sql.shuffle.partitions=32
```

**Cost:** ~$2.78/hour | **Runtime:** 3-8 hours | **Total:** $8-22

### Option B: High-Performance Cluster

Faster processing but higher cost.

```bash
gcloud dataproc clusters create emoarc-cluster-fast \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n2-highmem-16 \
  --master-boot-disk-size=100GB \
  --num-workers=8 \
  --worker-machine-type=n2-highmem-8 \
  --worker-boot-disk-size=100GB \
  --image-version=2.2-debian12 \
  --max-idle=30m \
  --enable-component-gateway \
  --properties=^#^spark:spark.driver.memory=48g#spark:spark.executor.memory=24g#spark:spark.executor.cores=4#spark:spark.sql.shuffle.partitions=64
```

**Cost:** ~$5/hour | **Runtime:** 1.5-4 hours | **Total:** $7.50-20

### Option C: Budget Cluster

Slower but cheaper, good for testing.

```bash
gcloud dataproc clusters create emoarc-cluster-budget \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n2-standard-4 \
  --master-boot-disk-size=50GB \
  --num-workers=2 \
  --worker-machine-type=n2-standard-4 \
  --worker-boot-disk-size=50GB \
  --image-version=2.2-debian12 \
  --max-idle=30m \
  --enable-component-gateway \
  --properties=^#^spark:spark.driver.memory=8g#spark:spark.executor.memory=8g#spark:spark.executor.cores=2
```

**Cost:** ~$0.80/hour | **Runtime:** 10-20 hours | **Total:** $8-16

---

## Step 3: Submit Processing Job

```bash
gcloud dataproc jobs submit pyspark \
  gs://YOUR-BUCKET-NAME/code/main.py \
  --cluster=emoarc-cluster \
  --region=us-central1 \
  --py-files=gs://YOUR-BUCKET-NAME/code/emoarc-pipeline.tar.gz \
  -- \
  --books-dir=gs://YOUR-BUCKET-NAME/data/books \
  --metadata=gs://YOUR-BUCKET-NAME/data/gutenberg_metadata.csv \
  --emotion-lexicon=gs://YOUR-BUCKET-NAME/data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
  --vad-lexicon=gs://YOUR-BUCKET-NAME/data/NRC-VAD-Lexicon-v2.1.txt \
  --output=gs://YOUR-BUCKET-NAME/output \
  --language=en \
  --num-chunks=20 \
  --num-topics=5 \
  --mode=cluster
```

**Monitor Job:**
```bash
# View job status
gcloud dataproc jobs describe JOB_ID --region=us-central1

# View logs (live)
gcloud dataproc jobs wait JOB_ID --region=us-central1

# View in web console
open https://console.cloud.google.com/dataproc/jobs/
```

---

## Step 4: Download Results

```bash
# List output files
gsutil ls -lh gs://YOUR-BUCKET-NAME/output/trajectories/

# Download trajectories CSV
mkdir -p outputs
gsutil -m cp -r gs://YOUR-BUCKET-NAME/output/trajectories ./outputs/

# Load into Django
cd web
python manage.py migrate
python manage.py load_books ../outputs/trajectories/
```

---

## Step 5: Cleanup

```bash
# Delete cluster (saves money)
gcloud dataproc clusters delete emoarc-cluster --region=us-central1

# Keep data in GCS for future use, or delete to save storage costs
# gsutil rm -r gs://YOUR-BUCKET-NAME/output
# gsutil rm -r gs://YOUR-BUCKET-NAME
```

---

## Monitoring & Debugging

### View Spark UI

```bash
# Get master node name
gcloud dataproc clusters describe emoarc-cluster \
  --region=us-central1 \
  --format='value(config.masterConfig.instanceNames[0])'

# SSH tunnel to Spark UI
gcloud compute ssh MASTER_NODE_NAME \
  --zone=us-central1-a \
  -- -L 8088:localhost:8088 -L 18080:localhost:18080

# Open in browser
open http://localhost:8088  # YARN ResourceManager
open http://localhost:18080  # Spark History Server
```

### Check Logs

```bash
# View job logs
gcloud dataproc jobs describe JOB_ID --region=us-central1

# Download driver logs
gsutil cp gs://dataproc-staging-REGION-PROJECT/google-cloud-dataproc-metainfo/CLUSTER/jobs/JOB_ID/driveroutput .
```

### Common Issues

**OutOfMemoryError:**
- Increase cluster size (more workers or larger machines)
- Increase Spark memory: `spark:spark.driver.memory=48g`
- Process in batches using `--limit` parameter

**Slow Performance:**
- Check shuffle partitions: `spark:spark.sql.shuffle.partitions=64`
- Enable adaptive query execution (already on by default)
- Use high-memory machines instead of standard

**File Not Found:**
- Verify GCS paths with `gsutil ls`
- Check bucket permissions
- Ensure data was uploaded correctly

---

## Cost Optimization Tips

1. **Use Preemptible Workers** (60-70% cheaper, but can be interrupted):
   ```bash
   --num-preemptible-workers=4 \
   --preemptible-worker-boot-disk-size=50GB
   ```

2. **Auto-scaling** (automatically adds/removes workers):
   ```bash
   --enable-component-gateway \
   --autoscaling-policy=POLICY_NAME
   ```

3. **Spot Instances** (even cheaper than preemptible):
   ```bash
   --worker-machine-type=n2-highmem-8 \
   --secondary-worker-type=non-preemptible
   ```

4. **Delete Cluster After Job:**
   ```bash
   --max-idle=10m  # Auto-delete after 10 min idle
   ```

---

## Alternative: AWS EMR

If you prefer AWS, here's the equivalent setup:

```bash
# Create EMR cluster
aws emr create-cluster \
  --name "EmoArc Pipeline" \
  --release-label emr-6.15.0 \
  --applications Name=Spark \
  --instance-type m5.2xlarge \
  --instance-count 5 \
  --use-default-roles \
  --auto-terminate

# Submit job
aws emr add-steps \
  --cluster-id j-XXXXXXXXXXXXX \
  --steps Type=Spark,Name="EmoArc",ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,s3://YOUR-BUCKET/main.py,--books-dir,s3://YOUR-BUCKET/data/books,...]
```

**AWS Cost:** Similar to GCP (~$2-3/hour for comparable cluster)

---

## Next Steps After Processing

1. **Download trajectories CSV** from GCS
2. **Load into Django** with `python manage.py load_books`
3. **Verify data quality:**
   - Check number of books loaded
   - Spot-check emotion scores
   - Verify LDA topics are present
   - Test bookshelves filtering
4. **Deploy Django app** (see deployment guide)
5. **Delete GCS bucket** (optional, to save storage costs)

---

## Summary

| Cluster Type | vCPUs | RAM | Cost/hr | Runtime | Total Cost |
|-------------|-------|-----|---------|---------|------------|
| Budget      | 12    | 48GB | $0.80 | 10-20h | $8-16 |
| Balanced    | 40    | 320GB | $2.78 | 3-8h | $8-22 |
| High-Perf   | 72    | 576GB | $5.00 | 1.5-4h | $7.50-20 |

**Recommendation:** Start with **Balanced** cluster for best cost/performance trade-off.

---

**Questions?** Check the main [README.md](../README.md) or open an issue.

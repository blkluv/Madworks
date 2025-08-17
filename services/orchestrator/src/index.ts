import 'dotenv/config';
import { Worker, QueueEvents, Job } from 'bullmq';
import IORedis from 'ioredis';
import { Pool } from 'pg';
import fs from 'fs';
import path from 'path';

const redis = new IORedis({
  port: Number(process.env.REDIS_PORT || 6379),
  host: process.env.REDIS_HOST || '127.0.0.1',
  maxRetriesPerRequest: null, // Fix for BullMQ v4+
});
const queueName = 'jobs';
const queueEvents = new QueueEvents(queueName, { connection: redis as any });

const pg = new Pool({
  host: process.env.PGHOST,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  port: Number(process.env.PGPORT || 5432),
});

const pyBase = process.env.PY_PIPELINE_BASE || 'http://localhost:8010';

async function updateStatus(jobId: string, status: string) {
  await pg.query('update jobs set status=$2, updated_at=now() where id=$1', [jobId, status]);
}

async function saveOutputs(jobId: string, outputs: any[]) {
  await pg.query('update jobs set outputs=$2, updated_at=now() where id=$1', [jobId, JSON.stringify(outputs)]);
}

async function downloadImage(url: string, dest: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download image: ${url}`);
  const arrayBuffer = await res.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  fs.writeFileSync(dest, buffer);
}

const worker = new Worker(
  queueName,
  async (job: Job) => {
    const jobId = (job.data as any).jobId as string;
    const { rows } = await pg.query('select * from jobs where id=$1', [jobId]);
    const record = rows[0];
    if (!record) return;
    let tempImagePath = '';
    try {
      await updateStatus(jobId, 'analyzing');
      // Download image from MinIO (internal URL)
      tempImagePath = path.join('/tmp', `${jobId}_input`);
      // The API already stores the internal MinIO URL, so use it directly
      const internalUrl = record.input_image_url;
      console.log('Downloading from:', internalUrl);
      await downloadImage(internalUrl, tempImagePath);
      // Send image as file to /ingest-analyze
      const form = new FormData();
      const imageBuffer = fs.readFileSync(tempImagePath);
      form.append('image', new Blob([imageBuffer], { type: 'image/png' }), 'input.png');
      const ingestRes = await fetch(`${pyBase}/ingest-analyze`, { method: 'POST', body: form as any });
      if (!ingestRes.ok) {
        const errorText = await ingestRes.text();
        console.error('ingest-analyze failed:', ingestRes.status, errorText);
        throw new Error(`ingest-analyze failed: ${ingestRes.status} - ${errorText}`);
      }
      const analysis = await ingestRes.json();

      await updateStatus(jobId, 'copy_drafting');
      const copyInstructions = typeof record.copy_instructions === 'string'
        ? record.copy_instructions
        : JSON.stringify(record.copy_instructions || {});
      const copyRes = await fetch(`${pyBase}/copy`, {
        method: 'POST',
        body: JSON.stringify({
          copy_instructions: copyInstructions,
          facts: record.facts || {},
          brand_kit_id: record.brand_kit_id,
          constraints: record.constraints || {}
        }),
        headers: { 'content-type': 'application/json' }
      });
      if (!copyRes.ok) throw new Error('copy failed');
      const copy = await copyRes.json();

      await updateStatus(jobId, 'composing');
      // Use the first crop proposal for now
      const crop_info = analysis.crops && analysis.crops[0] ? analysis.crops[0] : { width: 1080, height: 1080 };
      const composeRes = await fetch(`${pyBase}/compose`, {
        method: 'POST',
        body: JSON.stringify({
          copy,
          analysis,
          crop_info
        }),
        headers: { 'content-type': 'application/json' }
      });
      if (!composeRes.ok) throw new Error('compose failed');
      const composition = await composeRes.json();

      await updateStatus(jobId, 'rendering');
      const renderRes = await fetch(`${pyBase}/render`, {
        method: 'POST',
        body: JSON.stringify({
          composition,
          crop_info,
          job_id: jobId
        }),
        headers: { 'content-type': 'application/json' }
      });
      if (!renderRes.ok) throw new Error('render failed');
      const render = await renderRes.json();

      await updateStatus(jobId, 'qa');
      const qaRes = await fetch(`${pyBase}/qa`, {
        method: 'POST',
        body: JSON.stringify({
          composition,
          render,
          copy
        }),
        headers: { 'content-type': 'application/json' }
      });
      const qa = await qaRes.json();
      if (!qa.ok) throw new Error('QA failed: ' + (qa.error || 'unknown'));

      await updateStatus(jobId, 'exporting');
      const exportRes = await fetch(`${pyBase}/export`, {
        method: 'POST',
        body: JSON.stringify({
          render,
          job_id: jobId,
          timestamp: new Date().toISOString()
        }),
        headers: { 'content-type': 'application/json' }
      });
      if (!exportRes.ok) throw new Error('export failed');
      const exported = await exportRes.json();

      await saveOutputs(jobId, exported.outputs || []);
      await updateStatus(jobId, 'done');
    } catch (err) {
      console.error('Pipeline failed', err);
      await pg.query('update jobs set status=$2, errors = coalesce(errors, ' + "'[]'" + ') || $3::jsonb, updated_at=now() where id=$1', [jobId, 'failed', JSON.stringify([{ message: String(err) }])]);
      throw err;
    } finally {
      // Clean up temp image
      if (tempImagePath && fs.existsSync(tempImagePath)) {
        fs.unlinkSync(tempImagePath);
      }
    }
  },
  { connection: redis as any, concurrency: 2 }
);

queueEvents.on('failed', ({ jobId, failedReason }) => {
  console.error('Job failed', jobId, failedReason);
});

console.log('Worker started');



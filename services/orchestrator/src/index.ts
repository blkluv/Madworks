import 'dotenv/config';
import { Worker, QueueEvents, Job } from 'bullmq';
import IORedis from 'ioredis';
import { Pool } from 'pg';
import { request } from 'undici';

const redis = new IORedis(Number(process.env.REDIS_PORT || 6379), process.env.REDIS_HOST || '127.0.0.1');
const queueName = 'jobs';
const queueEvents = new QueueEvents(queueName, { connection: redis as any });

const pg = new Pool({
  host: process.env.PGHOST,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  port: Number(process.env.PGPORT || 5432),
});

const pyBase = process.env.PY_PIPELINE_BASE || 'http://localhost:8000';

async function updateStatus(jobId: string, status: string) {
  await pg.query('update jobs set status=$2, updated_at=now() where id=$1', [jobId, status]);
}

async function saveOutputs(jobId: string, outputs: any[]) {
  await pg.query('update jobs set outputs=$2, updated_at=now() where id=$1', [jobId, JSON.stringify(outputs)]);
}

const worker = new Worker(
  queueName,
  async (job: Job) => {
    const jobId = (job.data as any).jobId as string;
    const { rows } = await pg.query('select * from jobs where id=$1', [jobId]);
    const record = rows[0];
    if (!record) return;
    try {
      await updateStatus(jobId, 'analyzing');
      await request(`${pyBase}/ingest-analyze`, { method: 'POST', body: JSON.stringify({ image_url: record.input_image_url }), headers: { 'content-type': 'application/json' } });

      await updateStatus(jobId, 'copy_drafting');
      const copyRes = await request(`${pyBase}/copy`, { method: 'POST', body: JSON.stringify({ copy: record.copy_instructions || {}, facts: record.facts || {}, brand_kit_id: record.brand_kit_id, constraints: record.constraints || {} }), headers: { 'content-type': 'application/json' } });
      const copy = await copyRes.body.json();

      await updateStatus(jobId, 'composing');
      const composeRes = await request(`${pyBase}/compose`, { method: 'POST', body: JSON.stringify({ template_id: record.template_id, copy }), headers: { 'content-type': 'application/json' } });
      const composition = await composeRes.body.json();

      await updateStatus(jobId, 'rendering');
      const renderRes = await request(`${pyBase}/render`, { method: 'POST', body: JSON.stringify({ composition }), headers: { 'content-type': 'application/json' } });
      const render = await renderRes.body.json();

      await updateStatus(jobId, 'qa');
      await request(`${pyBase}/qa`, { method: 'POST', body: JSON.stringify({ render }), headers: { 'content-type': 'application/json' } });

      await updateStatus(jobId, 'exporting');
      const exportRes = await request(`${pyBase}/export`, { method: 'POST', body: JSON.stringify({ render }), headers: { 'content-type': 'application/json' } });
      const exported = await exportRes.body.json();

      await saveOutputs(jobId, exported.outputs || []);
      await updateStatus(jobId, 'done');
    } catch (err) {
      console.error('Pipeline failed', err);
      await pg.query('update jobs set status=$2, errors = coalesce(errors, ' + "'[]'" + ') || $3::jsonb, updated_at=now() where id=$1', [jobId, 'failed', JSON.stringify([{ message: String(err) }])]);
      throw err;
    }
  },
  { connection: redis as any, concurrency: 2 }
);

queueEvents.on('failed', ({ jobId, failedReason }) => {
  console.error('Job failed', jobId, failedReason);
});

console.log('Worker started');



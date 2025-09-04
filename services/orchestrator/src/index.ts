import 'dotenv/config';
import { Worker, QueueEvents, Job } from 'bullmq';
import IORedis from 'ioredis';
import { Pool } from 'pg';
import fs from 'fs';
import path from 'path';
import pino from 'pino';

const redis = new IORedis({
  port: Number(process.env.REDIS_PORT || 6379),
  host: process.env.REDIS_HOST || '127.0.0.1',
  maxRetriesPerRequest: null, // Fix for BullMQ v4+
});
const queueName = 'jobs';
const queueEvents = new QueueEvents(queueName, { connection: redis as any });

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

const pg = new Pool({
  host: process.env.PGHOST,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  port: Number(process.env.PGPORT || 5432),
});

const pyBase = process.env.PY_PIPELINE_BASE || 'http://localhost:8010';

async function updateStatus(jobId: string, status: string) {
  logger.info({ jobId, status }, 'job:status');
  await pg.query('update jobs set status=$2, updated_at=now() where id=$1', [jobId, status]);
}

async function saveOutputs(jobId: string, outputs: any[]) {
  logger.info({ jobId, outputsCount: outputs?.length || 0 }, 'job:outputs:save');
  await pg.query('update jobs set outputs=$2, updated_at=now() where id=$1', [jobId, JSON.stringify(outputs)]);
}

async function downloadImage(url: string, dest: string) {
  const t0 = process.hrtime.bigint();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download image: ${url}`);
  const arrayBuffer = await res.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  fs.writeFileSync(dest, buffer);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  logger.info({ url, dest, bytes: buffer.length, durationMs: ms.toFixed(1) }, 'download:image:ok');
}

const worker = new Worker(
  queueName,
  async (job: Job) => {
    const jobId = (job.data as any).jobId as string;
    // For orchestrator runs, force SVG path and avoid GPT image generation per requirements
    const useGptImage = false;
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
      logger.info({ jobId, internalUrl }, 'ingest:download:start');
      await downloadImage(internalUrl, tempImagePath);
      // Send image as file to /ingest-analyze
      const form = new FormData();
      const imageBuffer = fs.readFileSync(tempImagePath);
      form.append('image', new Blob([imageBuffer], { type: 'image/png' }), 'input.png');
      const tIngest0 = process.hrtime.bigint();
      const ingestRes = await fetch(`${pyBase}/ingest-analyze`, { method: 'POST', body: form as any });
      if (!ingestRes.ok) {
        const errorText = await ingestRes.text().catch(() => '');
        logger.error({ jobId, status: ingestRes.status, errorText }, 'ingest:analyze:failed');
        throw new Error(`ingest-analyze failed: ${ingestRes.status} - ${errorText}`);
      }
      const analysis = await ingestRes.json();
      const ingestMs = Number(process.hrtime.bigint() - tIngest0) / 1e6;
      logger.info({ jobId, ingestMs: ingestMs.toFixed(1), crops: analysis?.crops?.length || 0, original_url: analysis?.original_url, original_url_internal: analysis?.original_url_internal }, 'ingest:analyze:ok');

      await updateStatus(jobId, 'copy_drafting');
      const copyInstructions = typeof record.copy_instructions === 'string'
        ? record.copy_instructions
        : JSON.stringify(record.copy_instructions || {});
      const tCopy0 = process.hrtime.bigint();
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
      if (!copyRes.ok) {
        const errorText = await copyRes.text().catch(() => '');
        logger.error({ jobId, status: copyRes.status, errorText }, 'copy:failed');
        throw new Error('copy failed');
      }
      const copy = await copyRes.json();
      const copyMs = Number(process.hrtime.bigint() - tCopy0) / 1e6;
      logger.info({ jobId, copyMs: copyMs.toFixed(1), copyFields: Object.keys(copy || {}) }, 'copy:ok');

      // Use the first crop proposal for now
      const crop_info = analysis.crops && analysis.crops[0] ? analysis.crops[0] : { width: 1080, height: 1080 };

      let composition: any = {};
      if (!useGptImage) {
        await updateStatus(jobId, 'composing');
        const tCompose0 = process.hrtime.bigint();
        const composeRes = await fetch(`${pyBase}/compose`, {
          method: 'POST',
          body: JSON.stringify({
            copy,
            analysis,
            crop_info
          }),
          headers: { 'content-type': 'application/json' }
        });
        if (!composeRes.ok) {
          const errorText = await composeRes.text().catch(() => '');
          logger.error({ jobId, status: composeRes.status, errorText }, 'compose:failed');
          throw new Error('compose failed');
        }
        composition = await composeRes.json();
        const composeMs = Number(process.hrtime.bigint() - tCompose0) / 1e6;
        logger.info({ jobId, composeMs: composeMs.toFixed(1), layers: Array.isArray(composition?.layers) ? composition.layers.length : undefined }, 'compose:ok');
      }

      await updateStatus(jobId, 'rendering');
      const tRender0 = process.hrtime.bigint();
      const renderRes = await fetch(`${pyBase}/render`, {
        method: 'POST',
        body: JSON.stringify({
          composition,
          copy,
          analysis,
          crop_info,
          job_id: jobId,
          // Explicitly force SVG rasterization and avoid GPT image generation
          force_svg: true,
          use_gpt_image: false,
        }),
        headers: { 'content-type': 'application/json' }
      });
      if (!renderRes.ok) {
        const errorText = await renderRes.text().catch(() => '');
        logger.error({ jobId, status: renderRes.status, errorText }, 'render:failed');
        throw new Error('render failed');
      }
      const render = await renderRes.json();
      const renderMs = Number(process.hrtime.bigint() - tRender0) / 1e6;
      logger.info({ jobId, renderMs: renderMs.toFixed(1), outputs: render?.outputs?.length || 0, thumbnail_url: render?.thumbnail_url }, 'render:ok');

      await updateStatus(jobId, 'qa');
      const tQa0 = process.hrtime.bigint();
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
      const qaMs = Number(process.hrtime.bigint() - tQa0) / 1e6;
      logger.info({ jobId, qaMs: qaMs.toFixed(1) }, 'qa:ok');

      await updateStatus(jobId, 'exporting');
      const tExport0 = process.hrtime.bigint();
      const exportRes = await fetch(`${pyBase}/export`, {
        method: 'POST',
        body: JSON.stringify({
          render,
          job_id: jobId,
          timestamp: new Date().toISOString()
        }),
        headers: { 'content-type': 'application/json' }
      });
      if (!exportRes.ok) {
        const errorText = await exportRes.text().catch(() => '');
        logger.error({ jobId, status: exportRes.status, errorText }, 'export:failed');
        throw new Error('export failed');
      }
      const exported = await exportRes.json();
      const exportMs = Number(process.hrtime.bigint() - tExport0) / 1e6;
      logger.info({ jobId, exportMs: exportMs.toFixed(1), outputs: exported?.outputs?.length || 0 }, 'export:ok');

      await saveOutputs(jobId, exported.outputs || []);
      await updateStatus(jobId, 'done');
    } catch (err) {
      logger.error({ jobId, err }, 'pipeline:failed');
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
  logger.error({ jobId, failedReason }, 'queue:failed');
});

logger.info({ queueName, pyBase }, 'worker:started');



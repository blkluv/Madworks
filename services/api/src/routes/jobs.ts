import { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { z } from 'zod';
import { db } from '../db/client';
import { Queue } from 'bullmq';
import IORedis from 'ioredis';
import { uploadBuffer, publicUrl } from '../utils/s3';

const redis = new IORedis(Number(process.env.REDIS_PORT || 6379), process.env.REDIS_HOST || '127.0.0.1');
const jobsQueue = new Queue('jobs', { connection: redis as any });

const createJobSchema = z.object({
  image_url: z.string().url().optional(),
  template_id: z.string(),
  brand_kit_id: z.string().optional(),
  copy: z
    .object({
      headline: z.string().optional(),
      sub: z.string().optional(),
      cta: z.string().optional(),
      tone: z.string().optional(),
    })
    .optional(),
  facts: z.record(z.any()).optional(),
  constraints: z.record(z.any()).optional(),
});

export async function jobsRoutes(app: FastifyInstance) {
  app.post('/jobs', async (req: FastifyRequest, reply: FastifyReply) => {
    const id = `jb_${Math.random().toString(36).slice(2, 6)}${Date.now().toString(36).slice(-2)}`;
    req.log.info({ id }, 'jobs:create:start');

    let body: any = req.body;
    let inputUrl: string | null = null;

    if ((req as any).isMultipart && (req as any).isMultipart()) {
      req.log.info({ id }, 'jobs:create:multipart');
      const file = await (req as any).file();
      if (!file) return reply.badRequest('file field required for multipart');
      const buf = await file.toBuffer();
      req.log.debug({ id, filename: file.filename, mimetype: file.mimetype, bytes: buf.length }, 'jobs:create:file:received');
      const key = `uploads/${id}/${file.filename}`;
      const bucket = process.env.S3_BUCKET_ASSETS || 'assets';
      await uploadBuffer({ bucket, key, contentType: file.mimetype, body: buf });
      req.log.info({ id, bucket, key, size: buf.length }, 'jobs:create:file:uploaded');
      inputUrl = publicUrl(process.env.S3_BUCKET_ASSETS || 'assets', key);
      // parse fields if provided
      const fields = file.fields || {};
      body = Object.fromEntries(Object.entries(fields).map(([k, v]: any) => [k, v.value]));
    }

    const parsed = createJobSchema.safeParse(body);
    if (!parsed.success) return reply.badRequest(parsed.error.message);
    const data = parsed.data;
    if (!inputUrl) inputUrl = data.image_url || null;
    req.log.info({ id, template_id: data.template_id, hasInputUrl: Boolean(inputUrl) }, 'jobs:create:validated');

    await db.query(
      `insert into jobs (id, status, input_image_url, template_id, brand_kit_id, copy_instructions, facts, constraints)
       values ($1,'queued',$2,$3,$4,$5,$6,$7)`,
      [id, inputUrl, data.template_id, data.brand_kit_id || null, data.copy || {}, data.facts || {}, data.constraints || {}]
    );
    req.log.info({ id }, 'jobs:create:db:inserted');

    await jobsQueue.add('process', { jobId: id }, { removeOnComplete: { count: 100 }, removeOnFail: { count: 100 } });
    req.log.info({ id }, 'jobs:create:queued');

    const base = process.env.PUBLIC_API_BASE || '';
    const poll = base ? `${base.replace(/\/$/, '')}/v1/jobs/${id}` : `/v1/jobs/${id}`;
    req.log.info({ id, poll }, 'jobs:create:done');
    return { job_id: id, status: 'queued', poll };
  });

  app.get('/jobs/:id', async (req: FastifyRequest, reply: FastifyReply) => {
    const { id } = req.params as { id: string };
    req.log.debug({ id }, 'jobs:poll:start');
    const { rows } = await db.query('select id, status, outputs, errors from jobs where id=$1', [id]);
    if (!rows[0]) return reply.notFound('job not found');
    req.log.debug({ id, status: rows[0].status, outputs: Array.isArray(rows[0].outputs) ? rows[0].outputs.length : 0 }, 'jobs:poll:ok');
    return rows[0];
  });
}



import Fastify from 'fastify';
import fastifyCors from '@fastify/cors';
import fastifySensible from '@fastify/sensible';
import multipart from '@fastify/multipart';
import { jobsRoutes } from './routes/jobs';
import { templatesRoutes } from './routes/templates';

export async function buildServer() {
  const server = Fastify({ logger: true });
  await server.register(fastifyCors, { origin: true });
  await server.register(fastifySensible);
  const uploadLimitMb = Number(process.env.UPLOAD_LIMIT_MB || 50);
  await server.register(multipart, { limits: { fileSize: uploadLimitMb * 1024 * 1024 } });

  server.get('/health', async () => ({ ok: true }));

  await server.register(templatesRoutes, { prefix: '/v1' });
  await server.register(jobsRoutes, { prefix: '/v1' });
  return server;
}



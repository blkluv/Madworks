import Fastify, { FastifyError, FastifyReply, FastifyRequest } from 'fastify';
import fastifyCors from '@fastify/cors';
import fastifySensible from '@fastify/sensible';
import multipart from '@fastify/multipart';
import { jobsRoutes } from './routes/jobs';
import { templatesRoutes } from './routes/templates';

export async function buildServer() {
  const server = Fastify({
    logger: {
      level: process.env.LOG_LEVEL || 'info',
      // Basic redaction of sensitive headers
      redact: ['req.headers.authorization', 'request.headers.authorization', 'headers.authorization'],
    },
  });
  await server.register(fastifyCors, { origin: true });
  await server.register(fastifySensible);
  const uploadLimitMb = Number(process.env.UPLOAD_LIMIT_MB || 50);
  await server.register(multipart, { limits: { fileSize: uploadLimitMb * 1024 * 1024 } });

  // Global logging hooks
  server.addHook('onRequest', async (req: FastifyRequest) => {
    (req as any)._start = process.hrtime.bigint();
    req.log.info({ reqId: req.id, method: req.method, url: req.url, ip: req.ip }, 'request:start');
  });

  server.addHook('preHandler', async (req: FastifyRequest) => {
    const ct = req.headers['content-type'];
    const cl = req.headers['content-length'];
    req.log.debug({ reqId: req.id, contentType: ct, contentLength: cl }, 'request:preHandler');
  });

  server.addHook('onResponse', async (req: FastifyRequest, reply: FastifyReply) => {
    const start = (req as any)._start as bigint | undefined;
    const durationMs = start ? Number(process.hrtime.bigint() - start) / 1e6 : undefined;
    req.log.info({ reqId: req.id, statusCode: reply.statusCode, durationMs }, 'request:end');
  });

  server.setErrorHandler((err: FastifyError, req: FastifyRequest, reply: FastifyReply) => {
    req.log.error({ reqId: req.id, err }, 'request:error');
    reply.send(err);
  });

  server.get('/health', async () => ({
    ok: true,
    service: 'api',
    time: new Date().toISOString(),
    log_level: process.env.LOG_LEVEL || 'info',
  }));

  await server.register(templatesRoutes, { prefix: '/v1' });
  await server.register(jobsRoutes, { prefix: '/v1' });
  server.log.info('API server initialized and routes registered');
  return server;
}



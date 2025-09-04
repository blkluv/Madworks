import { FastifyInstance } from 'fastify';
import { db } from '../db/client';

export async function templatesRoutes(app: FastifyInstance) {
  app.get('/templates', async () => {
    app.log.debug('templates:list:start');
    const { rows } = await db.query('select id, name, format, preview_url from templates order by name');
    app.log.debug({ count: rows.length }, 'templates:list:ok');
    return rows;
  });
}



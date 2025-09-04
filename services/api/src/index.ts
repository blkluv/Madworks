import 'dotenv/config';
import { buildServer } from './server';

async function main() {
  const server = await buildServer();
  const port = Number(process.env.PORT || 3001);
  const host = '0.0.0.0';
  server.log.info({ host, port }, 'api:listening:start');
  await server.listen({ port, host });
}

main();



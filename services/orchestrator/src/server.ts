import http from 'http';
import sharp from 'sharp';

const PORT = Number(process.env.ORCH_RENDER_PORT || 8020);

function readJson(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => {
      data += chunk;
      // Basic guard: 32MB body limit
      if (data.length > 32 * 1024 * 1024) {
        reject(new Error('Payload too large'));
        req.destroy();
      }
    });
    req.on('end', () => {
      try {
        const json = JSON.parse(data || '{}');
        resolve(json);
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  // CORS for local dev convenience
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method === 'POST' && req.url === '/render-svg') {
    try {
      const body = await readJson(req);
      const svg: string = String(body.svg || '');
      const width = body.width ? Number(body.width) : undefined;
      const height = body.height ? Number(body.height) : undefined;
      if (!svg || svg.length < 20) {
        res.statusCode = 400;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ error: 'Missing or invalid svg' }));
        return;
      }
      // Render SVG -> PNG
      let pipeline = sharp(Buffer.from(svg));
      if (width || height) {
        pipeline = pipeline.resize({ width, height, fit: 'fill' });
      }
      const png = await pipeline.png().toBuffer();
      res.statusCode = 200;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ png_b64: png.toString('base64') }));
    } catch (err: any) {
      res.statusCode = 500;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: String(err && err.message || err) }));
    }
    return;
  }

  res.statusCode = 404;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({ error: 'Not found' }));
});

server.listen(PORT, () => {
  console.log(`Sharp render server listening on http://localhost:${PORT}`);
});

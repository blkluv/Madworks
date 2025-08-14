import AWS from 'aws-sdk';

const endpoint = process.env.S3_ENDPOINT;
const s3 = new AWS.S3({
  endpoint,
  region: process.env.S3_REGION || 'us-east-1',
  accessKeyId: process.env.S3_ACCESS_KEY,
  secretAccessKey: process.env.S3_SECRET_KEY,
  s3ForcePathStyle: true,
  signatureVersion: 'v4',
});

export async function uploadBuffer(params: {
  bucket: string;
  key: string;
  contentType: string;
  body: Buffer;
  cacheControl?: string;
}) {
  await s3
    .putObject({
      Bucket: params.bucket,
      Key: params.key,
      Body: params.body,
      ContentType: params.contentType,
      CacheControl: params.cacheControl || 'public, max-age=31536000, immutable',
      ACL: 'public-read',
    })
    .promise();
}

export function publicUrl(bucket: string, key: string): string {
  const base = process.env.PUBLIC_CDN_BASE || endpoint || '';
  if (base.includes('http')) {
    if (base.includes('amazonaws.com')) {
      return `${base.replace(/\/$/, '')}/${bucket}/${encodeURI(key)}`;
    }
    return `${base.replace(/\/$/, '')}/${bucket}/${encodeURI(key)}`;
  }
  return `/${bucket}/${encodeURI(key)}`;
}



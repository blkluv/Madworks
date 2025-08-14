import { Pool } from 'pg';

const pool = new Pool({
  host: process.env.PGHOST,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  port: Number(process.env.PGPORT || 5432),
});

export const db = {
  query: (text: string, params?: any[]) => pool.query(text, params),
};



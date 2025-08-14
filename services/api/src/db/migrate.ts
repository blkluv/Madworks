import 'dotenv/config';
import { db } from './client';

async function main() {
  await db.query(`
    create table if not exists templates (
      id text primary key,
      name text not null,
      format text not null,
      layout_svg text not null,
      text_slots jsonb not null default '[]',
      image_slots jsonb not null default '[]',
      safe_area jsonb,
      fonts jsonb not null default '[]',
      color_tokens jsonb not null default '[]',
      preview_url text
    );

    create table if not exists brand_kits (
      id text primary key,
      name text,
      primary_colors jsonb,
      secondary_colors jsonb,
      fonts jsonb,
      logo_url text,
      tone text
    );

    do $$ begin
      if not exists (select 1 from pg_type where typname = 'job_status') then
        create type job_status as enum ('queued','analyzing','copy_drafting','composing','rendering','qa','exporting','done','failed');
      end if;
    end $$;

    create table if not exists jobs (
      id text primary key,
      status job_status not null,
      input_image_url text,
      template_id text references templates(id),
      brand_kit_id text references brand_kits(id),
      copy_instructions jsonb,
      facts jsonb,
      constraints jsonb,
      outputs jsonb default '[]',
      errors jsonb default '[]',
      created_at timestamptz not null default now(),
      updated_at timestamptz not null default now()
    );
  `);

  const { rows } = await db.query('select count(*)::int as c from templates');
  if (rows[0]?.c === 0) {
    await db.query(
      `insert into templates (id,name,format,layout_svg,text_slots,image_slots,fonts,color_tokens,preview_url)
       values ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
      [
        'story-bold-plate-v3',
        'Story Bold Plate v3',
        'story',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1920"></svg>',
        JSON.stringify([{ id: 'headline', x: 80, y: 200, w: 920, h: 400 }]),
        JSON.stringify([{ id: 'photo', x: 0, y: 0, w: 1080, h: 1920 }]),
        JSON.stringify(['Inter', 'Archivo Black']),
        JSON.stringify(['primary', 'secondary']),
        null,
      ]
    );
  }

  console.log('DB migrated');
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});



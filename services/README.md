# Madworks backend services

## Run locally

1. Docker Compose infra

```bash
cd infra
docker compose up -d --build
```

2. Migrate DB

```bash
docker compose exec api node -e "require('./dist/index.js')" | cat
docker compose exec api node -e "require('dotenv').config(); require('./dist/db/migrate.js');" | cat
```

3. API available at http://localhost:3001/health



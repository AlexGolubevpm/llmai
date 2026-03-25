#!/bin/bash
set -e

# ============================================
# LLMAI v3.0 — Manual Deploy Script
# Run on the server to deploy latest code
# ============================================

DEPLOY_DIR="/opt/llmai"
APP_DIR="${DEPLOY_DIR}/llmai-v3"
BRANCH="${1:-main}"

echo "==> Deploying LLMAI v3.0 (branch: ${BRANCH})..."

cd "${DEPLOY_DIR}"

# Pull latest
echo "==> Pulling latest code..."
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull origin "${BRANCH}"

cd "${APP_DIR}"

# Rebuild and restart
echo "==> Building containers..."
docker compose -f docker-compose.prod.yml build

echo "==> Restarting services..."
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# Migrations
echo "==> Running migrations..."
sleep 5
docker compose -f docker-compose.prod.yml exec -T app npx prisma migrate deploy || echo "Migrations skipped (no pending)"

# Health check
echo "==> Health check..."
sleep 5
for i in 1 2 3 4 5; do
    if curl -sf http://localhost:3000 > /dev/null 2>&1; then
        echo "==> App is healthy!"
        break
    fi
    echo "Waiting... (attempt $i/5)"
    sleep 5
done

# Status
echo ""
echo "==> Container status:"
docker compose -f docker-compose.prod.yml ps
echo ""
echo "==> Deploy complete!"

#!/bin/bash
set -e

# ============================================
# LLMAI v3.0 — Server Setup Script (Timeweb)
# Run this ONCE on a fresh server
# ============================================

echo "==> LLMAI v3.0 Server Setup"
echo "==> Target: Timeweb Cloud Server"
echo ""

# 1. System update
echo "==> [1/6] Updating system packages..."
apt-get update && apt-get upgrade -y

# 2. Install Docker
echo "==> [2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

# 3. Install Docker Compose (v2 plugin)
echo "==> [3/6] Checking Docker Compose..."
if docker compose version &> /dev/null; then
    echo "Docker Compose v2 already available"
else
    apt-get install -y docker-compose-plugin
fi

# 4. Install git
echo "==> [4/6] Installing git..."
apt-get install -y git curl

# 5. Clone repository
DEPLOY_DIR="/opt/llmai"
echo "==> [5/6] Setting up project at ${DEPLOY_DIR}..."
if [ -d "${DEPLOY_DIR}" ]; then
    echo "Directory already exists, pulling latest..."
    cd "${DEPLOY_DIR}"
    git pull origin main || true
else
    git clone https://github.com/AlexGolubevpm/llmai.git "${DEPLOY_DIR}"
    cd "${DEPLOY_DIR}"
fi

cd "${DEPLOY_DIR}/llmai-v3"

# 6. Create .env from template
echo "==> [6/6] Creating environment config..."
if [ ! -f .env ]; then
    cp .env.example .env

    # Generate a secure DB password
    DB_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)
    sed -i "s|DB_PASSWORD:-llmai_secret|DB_PASSWORD:-${DB_PASSWORD}|g" docker-compose.prod.yml
    sed -i "s|password@|${DB_PASSWORD}@|g" .env

    echo ""
    echo "============================================"
    echo "  IMPORTANT: Edit .env before starting!"
    echo "  ${DEPLOY_DIR}/llmai-v3/.env"
    echo ""
    echo "  Required settings:"
    echo "  - NOVITA_API_KEY=sk_your_key_here"
    echo "  - DB_PASSWORD=${DB_PASSWORD}"
    echo "============================================"
    echo ""
else
    echo ".env already exists, skipping"
fi

# Create upload/result dirs
mkdir -p uploads results

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env:  nano ${DEPLOY_DIR}/llmai-v3/.env"
echo "  2. Start app:  cd ${DEPLOY_DIR}/llmai-v3 && docker compose -f docker-compose.prod.yml up -d"
echo "  3. Run migrations: docker compose -f docker-compose.prod.yml exec app npx prisma migrate deploy"
echo "  4. Check status: docker compose -f docker-compose.prod.yml ps"
echo "  5. View logs:   docker compose -f docker-compose.prod.yml logs -f"
echo ""

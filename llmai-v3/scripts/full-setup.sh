#!/bin/bash
set -e

# ============================================
# LLMAI — Full Server + CI/CD Setup
# Run this on Timeweb server as root
# ============================================

echo "==> LLMAI Full Setup Starting..."
echo ""

# ---- 1. System ----
echo "==> [1/7] Updating system..."
apt-get update -qq && apt-get upgrade -y -qq

# ---- 2. Docker ----
echo "==> [2/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi
echo "Docker: $(docker --version)"

# ---- 3. Docker Compose ----
echo "==> [3/7] Checking Docker Compose..."
if ! docker compose version &> /dev/null; then
    apt-get install -y -qq docker-compose-plugin
fi
echo "Compose: $(docker compose version)"

# ---- 4. Git + tools ----
echo "==> [4/7] Installing tools..."
apt-get install -y -qq git curl

# ---- 5. SSH deploy key ----
echo "==> [5/7] Setting up deploy SSH key..."
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Add deploy key (used by GitHub Actions)
DEPLOY_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHOQiGgafsx20/u8/hhpiIG5tAJZiwE+77INrAN2whsp llmai-deploy"

if ! grep -q "llmai-deploy" /root/.ssh/authorized_keys 2>/dev/null; then
    echo "$DEPLOY_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "Deploy key added to authorized_keys"
else
    echo "Deploy key already present"
fi

# ---- 6. Clone repo ----
echo "==> [6/7] Cloning repository..."
DEPLOY_DIR="/opt/llmai"

if [ -d "${DEPLOY_DIR}/.git" ]; then
    echo "Repo already exists, pulling latest..."
    cd "${DEPLOY_DIR}"
    git pull origin main || git pull origin claude/analyze-app-architecture-PcBp6 || true
else
    rm -rf "${DEPLOY_DIR}"
    git clone https://github.com/AlexGolubevpm/llmai.git "${DEPLOY_DIR}"
    cd "${DEPLOY_DIR}"
    git checkout claude/analyze-app-architecture-PcBp6
fi

cd "${DEPLOY_DIR}/llmai-v3"

# ---- 7. Create .env ----
echo "==> [7/7] Creating environment config..."

DB_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)

if [ ! -f .env ]; then
    cat > .env << ENVEOF
# Database
DATABASE_URL=postgresql://llmai:${DB_PASSWORD}@postgres:5432/llmai
DB_PASSWORD=${DB_PASSWORD}

# Redis
REDIS_URL=redis://redis:6379

# openrouter AI — CHANGE THIS!
OPENROUTER_API_KEY=sk_your_key_here
OPENROUTER_BASE_URL=https://api.openrouter.ai/openai
# removed=60

# WD Tagger
WD_TAGGER_URL=https://deepghs-wd-tagger-heatmap-more-models.hf.space

# File storage
UPLOAD_DIR=./uploads
RESULT_DIR=./results

# App
NEXT_PUBLIC_APP_URL=http://$(hostname -I | awk '{print $1}'):3000
NODE_ENV=production
ENVEOF
    echo ".env created with DB_PASSWORD=${DB_PASSWORD}"
else
    echo ".env already exists, keeping it"
fi

# Create dirs
mkdir -p uploads results

# ---- Start containers ----
echo ""
echo "==> Building and starting containers..."
docker compose -f docker-compose.prod.yml up -d --build

# Wait for health
echo "==> Waiting for services..."
sleep 15

# Show status
echo ""
echo "============================================"
echo "  SETUP COMPLETE!"
echo "============================================"
echo ""
echo "  Server IP:    $(hostname -I | awk '{print $1}')"
echo "  App URL:      http://$(hostname -I | awk '{print $1}'):3000"
echo "  Deploy dir:   ${DEPLOY_DIR}/llmai-v3"
echo "  DB Password:  ${DB_PASSWORD}"
echo ""
echo "  Container status:"
docker compose -f docker-compose.prod.yml ps
echo ""
echo "  NEXT STEPS:"
echo "  1. Edit openrouter API key:"
echo "     nano ${DEPLOY_DIR}/llmai-v3/.env"
echo "     (change OPENROUTER_API_KEY=sk_your_key_here)"
echo ""
echo "  2. Restart after edit:"
echo "     cd ${DEPLOY_DIR}/llmai-v3"
echo "     docker compose -f docker-compose.prod.yml restart"
echo ""
echo "  3. Add GitHub Secrets (see below)"
echo "============================================"
echo ""
echo "  GITHUB SECRETS TO ADD:"
echo "  Repository: github.com/AlexGolubevpm/llmai"
echo "  Settings -> Secrets and variables -> Actions -> New repository secret"
echo ""
echo "  TIMEWEB_HOST = $(hostname -I | awk '{print $1}')"
echo "  TIMEWEB_USER = root"
echo "  TIMEWEB_PORT = 22"
echo "  TIMEWEB_SSH_KEY = (paste the private key below)"
echo ""
echo "--- PRIVATE KEY START (copy everything between the dashes) ---"
cat << 'KEYEOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACBzkIhoGn7MdtP7vP4YaYiBubQCWYsBPu+yDawDdsIbKQAAAJAvcMKLL3DC
iwAAAAtzc2gtZWQyNTUxOQAAACBzkIhoGn7MdtP7vP4YaYiBubQCWYsBPu+yDawDdsIbKQ
AAAEDtrehFQKPZvBF2DKWmxQWAvfdBqgBn0RRDsV6fTSarVXOQiGgafsx20/u8/hhpiIG5
tAJZiwE+77INrAN2whspAAAADGxsbWFpLWRlcGxveQE=
-----END OPENSSH PRIVATE KEY-----
KEYEOF
echo "--- PRIVATE KEY END ---"
echo ""

#!/usr/bin/env bash
# =============================================================================
# Polymarket BTC Bot — Server Setup Script
# =============================================================================
# Provisions a fresh Ubuntu 22.04+ server for production deployment.
#
# Usage:
#   chmod +x deploy/setup_server.sh
#   sudo ./deploy/setup_server.sh
# =============================================================================

set -euo pipefail

# --- Configuration ---
APP_DIR="/opt/btc-bot"
APP_USER="deploy"
PYTHON_VERSION="3.12"
REPO_URL="${1:-}"  # Pass your git repo URL as first argument

echo "============================================"
echo "Polymarket BTC Bot — Server Setup"
echo "============================================"

# --- 1. System Updates ---
echo "[1/8] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# --- 2. Install Dependencies ---
echo "[2/8] Installing system dependencies..."
apt-get install -y -qq \
    software-properties-common \
    git \
    curl \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev

# --- 3. Install Python ---
echo "[3/8] Installing Python ${PYTHON_VERSION}..."
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-venv" \
    "python${PYTHON_VERSION}-dev"

# --- 4. Create Application User ---
echo "[4/8] Creating application user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /bin/bash "$APP_USER"
    echo "  Created user: $APP_USER"
else
    echo "  User $APP_USER already exists"
fi

# --- 5. Clone Repository ---
echo "[5/8] Setting up application directory..."
mkdir -p "$APP_DIR"
if [ -n "$REPO_URL" ]; then
    if [ -d "$APP_DIR/.git" ]; then
        echo "  Repository exists, pulling latest..."
        cd "$APP_DIR" && git pull
    else
        echo "  Cloning repository..."
        git clone "$REPO_URL" "$APP_DIR"
    fi
else
    echo "  [!] No REPO_URL provided. Clone manually:"
    echo "      git clone <your-repo> $APP_DIR"
fi

# --- 6. Python Virtual Environment ---
echo "[6/8] Setting up Python virtual environment..."
cd "$APP_DIR"
"python${PYTHON_VERSION}" -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
deactivate

# --- 7. Set Permissions ---
echo "[7/8] Setting file permissions..."
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 "$APP_DIR/.env" 2>/dev/null || echo "  [!] .env not found — copy .env.example and fill in values"

# --- 8. Install systemd Service ---
echo "[8/8] Installing systemd service..."
cp "$APP_DIR/deploy/polymarket_bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable polymarket_bot.service

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Copy and configure .env:"
echo "     cp $APP_DIR/.env.example $APP_DIR/.env"
echo "     nano $APP_DIR/.env"
echo ""
echo "  2. Copy the trained model:"
echo "     scp data/models/lgbm_btc_5m.txt server:$APP_DIR/data/models/"
echo ""
echo "  3. Start the service:"
echo "     sudo systemctl start polymarket_bot"
echo ""
echo "  4. Monitor logs:"
echo "     journalctl -u polymarket_bot -f"
echo ""

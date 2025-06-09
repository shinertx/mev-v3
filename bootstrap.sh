#!/usr/bin/env bash
# ---
# role: [infra]
# purpose: Bootstrap script
# dependencies: []
# mutation_ready: true
# test_status: [ci_passed]
# ---
set -euo pipefail

echo "=== MEV-OG / MEV-V3 BOOTSTRAP ==="

# 1. Check PROJECT_BIBLE.md presence and AGENTS.md compliance
if [[ ! -f "PROJECT_BIBLE.md" ]] || [[ ! -f "AGENTS.md" ]]; then
    echo "FATAL: PROJECT_BIBLE.md and/or AGENTS.md missing in repo root."
    exit 1
fi
echo "[\u2713] PROJECT_BIBLE.md and AGENTS.md present"

# 2. Check for GCP Secret Manager CLI (gcloud) & prompt user if missing
if ! command -v gcloud &>/dev/null; then
    echo "FATAL: gcloud CLI not found. Install GCP SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "[\u2713] gcloud CLI available"

# 3. Ensure required secrets are loaded (edit below for your project secrets)
REQUIRED_SECRETS=("MEVOG_API_KEY" "DB_URI")
for secret in "${REQUIRED_SECRETS[@]}"; do
    if ! gcloud secrets versions access latest --secret="$secret" &>/dev/null; then
        echo "FATAL: Secret $secret not found in GCP Secret Manager"
        exit 1
    fi
done
echo "[\u2713] All required secrets present in GCP Secret Manager"

# 4. Install Python dependencies if requirements.txt present
if [[ -f "requirements.txt" ]]; then
    echo "[*] Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# 5. Install Node.js dependencies if package.json present
if [[ -f "package.json" ]]; then
    echo "[*] Installing Node.js dependencies..."
    npm install
fi

# 6. Any project-specific bootstrap logic (add as needed)
echo "[*] Bootstrap complete. Ready for local/dev/test CI."

#!/usr/bin/env bash
# Lance l'application Streamlit (accessible depuis le réseau sur le port 8501).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate
exec streamlit run app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0

#!/usr/bin/env bash
# Installation complète sur une VM Ubuntu (testé sur Oracle Cloud, sans GPU).
# Usage : bash scripts/setup_vm.sh
set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3.2}"

echo "==> 1/4 Paquets système"
sudo apt update
sudo apt install -y python3-venv python3-pip git curl tmux

echo "==> 2/4 Ollama et modèle ${MODEL}"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull "${MODEL}"

echo "==> 3/4 Environnement Python"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "==> 4/4 Terminé"
echo "Lancer l'application :  bash scripts/run.sh"

# Déploiement sur une VM Oracle Cloud (LabCloud ENSET)

L'application est hébergée sur une **machine virtuelle Oracle Cloud (OCI)** fournie par la plateforme **LabCloud** de l'ENSET.

## Caractéristiques de la VM

| Élément | Valeur |
|---|---|
| Système | Ubuntu |
| Processeur | AMD EPYC 9J14 (Genoa), 4 cœurs / 8 vCPU |
| Mémoire | 16 Go |
| Disque | 200 Go |
| GPU | Aucun (inférence sur CPU) |
| Utilisateur | `ubuntu`, connexion SSH par clé |
| Port ouvert par défaut | 22/TCP (SSH) |
| Port ajouté | 8501/TCP (Streamlit) |

## 1. Créer la VM et se connecter

1. Dans LabCloud, ajouter sa **clé SSH publique**, puis créer et démarrer la machine.
2. Copier l'adresse IP publique et la commande SSH affichées :

```bash
ssh ubuntu@<IP_PUBLIQUE_DE_LA_VM>
```

L'IP publique peut changer lorsque la machine est arrêtée puis redémarrée : la relire dans LabCloud.

## 2. Installer le projet

```bash
git clone https://github.com/Fahdkrachel/enset-rag-chatbot.git
cd enset-rag-chatbot
bash scripts/setup_vm.sh
```

Le script installe les paquets système, **Ollama**, télécharge `llama3.2` (environ 2 Go), crée l'environnement virtuel Python et installe PyTorch (version CPU) et les dépendances.

## 3. Ouvrir le port 8501

Dans LabCloud, section **« Ports ouverts »** : protocole `TCP`, port `8501`, usage `Streamlit`, puis **Ouvrir**.

Si la page ne s'ouvre toujours pas, le pare-feu de l'image Ubuntu peut bloquer le port :

```bash
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```

## 4. Lancer l'application

**Option A : manuellement**

```bash
bash scripts/run.sh
```

Pour qu'elle continue après la fermeture du terminal SSH, la lancer dans `tmux` (`tmux new -s rag`, lancer le script, puis `Ctrl+B` suivi de `D`).

**Option B : service systemd (recommandé)** : l'application démarre toute seule avec la VM.

```bash
sudo cp deploy/enset-rag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now enset-rag
sudo systemctl status enset-rag
```

Adapter `User` et les chemins dans le fichier si le projet n'est pas dans `/home/ubuntu/enset-rag-chatbot`. Voir les journaux : `journalctl -u enset-rag -f`. Ollama est installé comme service et démarre aussi avec la VM.

Ouvrir ensuite `http://<IP_PUBLIQUE_DE_LA_VM>:8501`.

## Accès privé par tunnel SSH (sans ouvrir le port)

Streamlit n'a ni authentification ni HTTPS. Pour un usage privé, il est préférable de ne pas ouvrir le port 8501 et de passer par un tunnel SSH, depuis son PC :

```bash
ssh -L 8501:localhost:8501 ubuntu@<IP_PUBLIQUE_DE_LA_VM>
```

puis ouvrir `http://localhost:8501`.

## Mise à jour du code

```bash
cd ~/enset-rag-chatbot
git pull
sudo systemctl restart enset-rag      # si le service systemd est utilisé
```

## Dépannage

| Symptôme | Cause probable et solution |
|---|---|
| La page ne s'ouvre pas | Port 8501 non ouvert dans LabCloud, ou pare-feu de la VM (commande `iptables` ci-dessus), ou IP changée |
| Erreur de connexion sur le port 11434 | Ollama arrêté : `sudo systemctl start ollama`, puis `ollama list` |
| `model 'llama3.2' not found` | `ollama pull llama3.2` |
| Réponses très lentes | Inférence sur CPU : utiliser un modèle plus petit (`ollama pull llama3.2:1b`, puis le saisir dans « Modèle Ollama »), baisser « Candidats max à noter » |
| Le premier traitement du PDF est long | Téléchargement unique du modèle d'embedding (environ 500 Mo) |
| `ModuleNotFoundError` | Environnement virtuel non activé : `source venv/bin/activate` |

# MLflow CV Tiny (YOLO) - Projet MLOps

## 📋 Description du Projet

Ce projet démontre une pipeline **MLOps complète** pour l'entraînement et le suivi de modèles de détection d'objets utilisant **YOLOv8** et **MLflow**. L'objectif est de tracer, comparer et promouvoir plusieurs runs d'entraînement sur un mini-dataset COCO (classe unique : `person`) sur la plateforme MLflow.

### Caractéristiques principales
- ✅ Entraînement **YOLOv8 Nano** avec gridSearching sur hyperparamètres
- ✅ Suivi automatique des métriques et artefacts via **MLflow**
- ✅ Storage d'artefacts sur **MinIO** (S3-compatible)
- ✅ Scripts multi-plateforme (Bash, PowerShell, CMD)
- ✅ Décision de promotion basée sur comparative analysis
- ✅ Infrastructure **containerisée** (Docker Compose)

---

## 🚀 Démarrage Rapide

### 1. Installation des dépendances

```bash
# Cloner le repository
git clone <repo-url>
cd mlflow-cv-yolo-main

# Installer les dépendances Python
pip install -r requirements.txt
```

### 2. Lancer l'infrastructure MLflow

```bash
# Démarrer les services (MLflow + MinIO)
docker compose up -d

# Vérifier que les services sont actifs
docker compose ps
```

**Services disponibles :**
- **MLflow UI** : http://localhost:5000
- **MinIO Console** : http://localhost:9001 (user: `minio`, pass: `minio12345`)

### 3. Préparer le dataset

```bash
# Créer un mini-dataset COCO avec la classe "person" uniquement
python tools/make_tiny_person_from_coco128.py
```

Cela générera une structure de données dans le dossier `data/` :
```
data/
├── images/
│   ├── train/  (40 images)
│   └── val/    (10 images)
└── labels/     (annotations YOLO format)
```

### 4. Entraîner un modèle de base

```bash
# Entraînement simple (3 epochs, taille image 320x320)
python src/train_cv.py --epochs 3 --imgsz 320 --exp-name cv_yolo_tiny
```

### 5. (Optionnel) Lancer une grille d'expériences

**Linux / macOS :**
```bash
bash scripts/run_grid.sh
```

**Windows (PowerShell) :**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_grid.ps1
```

**Windows (CMD) :**
```cmd
scripts\run_grid.cmd
```

---

## 📁 Structure du Projet

```
mlflow-cv-yolo-main/
│
├── src/                           # Code source principal
│   ├── train_cv.py               # Script d'entraînement YOLO
│   ├── utils.py                  # Utilitaires (seeds, logging, etc.)
│   └── __init__.py
│
├── scripts/                       # Scripts d'exécution
│   ├── run_grid.sh              # Grid search (Linux/macOS)
│   ├── run_grid.ps1             # Grid search (PowerShell)
│   ├── run_grid.cmd             # Grid search (CMD)
│   ├── register_model.py        # Enregistrer un modèle en production
│   ├── analyze_runs.py          # Analyser et comparer les runs
│   ├── relog_metrics.py         # Re-logger les métriques
│   └── upload_and_register.py   # Upload et registration combinées
│
├── tools/                         # Utilitaires de données
│   └── make_tiny_person_from_coco128.py  # Générer mini-dataset
│
├── data/                          # Données
│   ├── tiny_coco.dvc            # DVC tracking
│   └── tiny_coco.yaml           # Config dataset
│
├── reports/                       # Rapports et analyses
│   ├── DECISION_PROMOTION.md    # Décision de promotion du meilleur modèle
│   ├── runs_analysis.csv        # Résumé des runs
│   └── templates/
│       └── decision_template.md # Template pour décisions
│
├── docker-compose.yml            # Configuration Docker (MLflow + MinIO)
├── Dockerfile.mlflow             # Dockerfile MLflow personnalisé
├── mlflow.env                    # Variables d'environnement MLflow
├── requirements.txt              # Dépendances Python
└── README.md                     # Cette documentation
```

---

## 🔧 Configuration et Variables d'Environnement

### mlflow.env
```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio12345
```

### requirements.txt
Dépendances clés :
- **mlflow** (≥2.10) : Tracking et registry des modèles
- **ultralytics** (≥8.1) : YOLOv8
- **opencv-python** : Traitement d'images
- **pandas**, **numpy**, **matplotlib** : Data science
- **requests** : Appels HTTP

---

## 📊 Workflow MLOps Typique

### Phase 1 : Expérimentation

```bash
# Lancer une série d'entraînements
python src/train_cv.py --epochs 3 --imgsz 320 --lr0 0.01
python src/train_cv.py --epochs 3 --imgsz 416 --lr0 0.01
python src/train_cv.py --epochs 5 --imgsz 320 --lr0 0.005
```

Consultez les résultats sur **MLflow UI** (http://localhost:5000) :
- Comparez les **métriques** : mAP@50, mAP50-95, precision, recall
- Visualisez les **artefacts** : images de résultats, matrices de confusion, poids

### Phase 2 : Analyse et Comparaison

```bash
# Générer un rapport comparatif
python scripts/analyze_runs.py
```

Voir [reports/runs_analysis.csv](reports/runs_analysis.csv) pour les résultats.

### Phase 3 : Décision de Promotion

Complétez [reports/DECISION_PROMOTION.md](reports/DECISION_PROMOTION.md) :
- Identifiez le meilleur run selon vos critères (mAP@50, latence, etc.)
- Documentez les alternatives considérées
- Justifiez votre choix

Exemple (déjà rempli) :
```markdown
## Candidat promu
- **Run ID** : 6eddc182
- **Paramètres** : epochs=3, imgsz=416, lr0=0.01, seed=42
- **Meilleure métrique** : mAP@50 = 0.3227
```

### Phase 4 : Enregistrement en Production

```bash
# Enregistrer le meilleur modèle dans MLflow Model Registry
python scripts/register_model.py \
    --run-id 6eddc182 \
    --model-name yolov8n_person_detector \
    --stage Production
```

---

## 🎯 Métriques Clés

| Métrique | Description |
|----------|-------------|
| **mAP@50** | 0.3227 (32.3%) |
| **mAP50-95** | 0.2728|
| **Precision** | 0.008 |
| **Recall** |0.7742 |


---

## 🐳 Gestion des Services Docker

### Démarrer l'infrastructure

```bash
docker compose up -d
```

### Vérifier l'état

```bash
docker compose ps
```

### Arrêter les services

```bash
docker compose down
```

### Nettoyer les volumes (WARNING : supprime les données)

```bash
docker compose down -v
```

### Consulter les logs

```bash
docker compose logs -f mlflow
docker compose logs -f minio
```

---

## 📈 Exemple : Grille de Recherche

Le script [scripts/run_grid.sh](scripts/run_grid.sh) lance une série d'entraînements avec différents hyperparamètres :

```bash
for epochs in 3 5; do
    for imgsz in 320 416; do
        for lr0 in 0.005 0.01; do
            for seed in 1 42; do
                python src/train_cv.py \
                    --epochs $epochs \
                    --imgsz $imgsz \
                    --lr0 $lr0 \
                    --seed $seed \
                    --exp-name "yolov8n_e${epochs}_sz${imgsz}_lr${lr0}_s${seed}"
            done
        done
    done
done
```

**Résultat** : 32 runs générés, permettant une analyse comparative exhaustive.

---

## 🔍 Exemple d'Analyse (Résumé du Projet)

D'après [DECISION_PROMOTION.md](reports/DECISION_PROMOTION.md) :

### Meilleur Run
- **ID** : `6eddc182` (yolov8n_e3_sz416_lr0.01_s42)
- **mAP@50** : 0.3227 ✅
- **Recall** : 0.7742 (détecte 77% des personnes)

### Insights
1. **Image size = 416px** améliore mAP de +17% vs 320px
2. **Learning rate 0.01** marge faible vs 0.005 (non significatif)
3. **Variance inter-seed** : ~2% (modèle stable)

---

## 🛠️ Scripts Utiles

### `analyze_runs.py`
Génère une CSV comparative de tous les runs :
```bash
python scripts/analyze_runs.py
```

### `register_model.py`
Enregistre un modèle dans MLflow Model Registry :
```bash
python scripts/register_model.py --run-id <run-id> --model-name <name> --stage Production
```

### `relog_metrics.py`
Re-logger les métriques d'un run existant :
```bash
python scripts/relog_metrics.py --run-id <run-id>
```

### `upload_and_register.py`
Combiner upload d'artefacts et enregistrement du modèle :
```bash
python scripts/upload_and_register.py --run-id <run-id>
```

---

## 🐛 Troubleshooting

### "Cannot connect to MLflow server"
```bash
# Vérifier que les services Docker sont actifs
docker compose ps

# Vérifier l'URL MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### "No module named 'ultralytics'"
```bash
pip install --upgrade ultralytics
```

### "MinIO bucket not created"
```bash
# Vérifier que minio-mc a complété son initialisation
docker compose logs minio-mc

# Récréer le bucket manuellement via MinIO Console : http://localhost:9001
```

### "CUDA out of memory"
```bash
# Réduire la taille du batch ou l'image
python src/train_cv.py --epochs 3 --imgsz 256 --batch 8
```

---

## 📚 Références

- **MLflow Documentation** : https://mlflow.org/docs/latest/index.html
- **YOLOv8 Docs** : https://docs.ultralytics.com/
- **MinIO S3 API** : https://min.io/docs/minio/linux/index.html
- **COCO Dataset** : https://cocodataset.org/

---

## 📝 Licence

À remplir selon vos besoins.

---

## 📧 Contact & Support

Pour toute question ou problème, veuillez :
1. Consulter les logs : `docker compose logs`
2. Vérifier le [Troubleshooting](#-troubleshooting)
3. Ouvrir une issue sur le repository

---

**Dernière mise à jour** : 17 janvier 2026


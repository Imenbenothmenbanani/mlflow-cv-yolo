# Décision de promotion — TP MLflow (CV YOLO Tiny)

## Objectifs et contraintes
- **Objectif principal** : Maximiser mAP@50 pour la détection de personnes sur mini-dataset COCO128
- **Contraintes** :
  - Modèle nano YOLOv8n (contrainte taille/vitesse)
  - Entraînement court (3 epochs) - exploratory phase
  - Dataset limité (40 images train, 1 classe)

## Candidat promu
- **Run name / ID** : `yolov8n_e3_sz416_lr0.01_s42` / `6eddc182`
- **Paramètres clés** :
  - epochs: 3
  - imgsz: 416
  - lr0: 0.01
  - seed: 42
- **Métriques** :
  - **mAP@50: 0.3227** ✅ (meilleur score)
  - mAP50-95: 0.2728
  - precision: 0.008 (très faible - dataset déséquilibré)
  - recall: 0.7742 (bon - détecte 77% des personnes)

## Comparaison (résumé)

### Alternative A : Image size 416 vs 320
- **Pour 416** : +17% de mAP@50 (0.30 vs 0.27), meilleur recall (+6.5%)
- **Contre 416** : Temps d'entraînement légèrement plus long, plus de mémoire
- **Verdict** : **416 pixels clairement supérieur**

### Alternative B : Learning rate 0.01 vs 0.005
- **Pour 0.01** : Légère amélioration (~1-2% mAP selon seed)
- **Contre 0.01** : Différence non significative (dans la variance seed)
- **Verdict** : **0.01 acceptable, mais impact faible**

### Alternative C : Seed 42 vs 1
- **Observation** : Variance de 2% sur mAP@50 entre seeds
- **Conclusion** : Variance acceptable, seed 42 systématiquement légèrement meilleur

### Observations générales
- **Variance inter-seed** : Faible (~2%), modèle stable
- **Convergence** : 3 epochs suffisants pour ce mini-dataset
- **Précision faible** : Problème lié au dataset déséquilibré (beaucoup de faux positifs)
- **Recall élevé** : Le modèle détecte bien les vraies personnes (77%)

## Risques et mitigations

### Risque 1 : Overfitting sur 40 images
- **Mitigation** : Ce modèle est exploratoire uniquement. Pour production, ré-entraîner sur COCO complet (118K images)

### Risque 2 : Précision de 0.008 inacceptable
- **Mitigation** : Implémenter post-processing (NMS tuning), data augmentation, équilibrage dataset

### Risque 3 : Seed dependency
- **Mitigation** : Variance faible (2%), acceptable. Pour production, moyenner plusieurs runs.

## Décision

- **Promouvoir** : **Oui** ✅
- **Justification** :
  - Meilleur mAP@50 des 9 runs (0.3227)
  - Configuration optimale identifiée (416px, lr=0.01)
  - Recall acceptable (77%) pour détection de personnes
  - Variance seed faible (stabilité)
  
- **Étapes suivantes** :
  1. ✅ Enregistrer dans MLflow Model Registry (stage Staging)
  2. Tester sur validation set complet COCO128 (non filtré)
  3. Si résultats satisfaisants → Transition vers Production
  4. Pour déploiement réel → Ré-entraîner sur COCO full dataset
  5. Optimiser précision via hyperparameter tuning (conf_threshold, iou_threshold)

---

**Date** : 2025-01-23  
**Auteur** : TP4 MLOps - Experiment Tracking

"""
Script pour uploader les poids locaux dans MLflow et enregistrer le modèle
"""
import os
import mlflow
from pathlib import Path

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "yolo_person_detector"

def upload_and_register():
    """Upload les poids locaux et enregistre le modèle"""
    
    print("🔍 Recherche du meilleur modèle...")
    print("=" * 60)
    
    # Chercher l'expérience
    experiment = mlflow.get_experiment_by_name("cv_yolo_tiny")
    if not experiment:
        print("❌ Expérience 'cv_yolo_tiny' non trouvée!")
        return
    
    # Trouver le meilleur run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and metrics.mAP50 > 0",
        order_by=["metrics.mAP50 DESC"],
        max_results=1
    )
    
    if runs.empty:
        print("❌ Aucun run trouvé!")
        return
    
    best_run = runs.iloc[0]
    run_id = best_run['run_id']
    run_name = best_run.get('tags.mlflow.runName', 'Unknown')
    mAP50 = best_run.get('metrics.mAP50', 0)
    
    print(f"✅ Meilleur run:")
    print(f"   Run: {run_name}")
    print(f"   ID: {run_id[:8]}...")
    print(f"   mAP@50: {mAP50:.4f}")
    print()
    
    # Chercher les poids locaux
    local_weights = Path(f"runs/{run_name}/weights/best.pt")
    
    if not local_weights.exists():
        print(f"❌ Poids non trouvés: {local_weights}")
        return
    
    print(f"📦 Poids trouvés: {local_weights}")
    print(f"   Taille: {local_weights.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    try:
        # Upload les artefacts dans MLflow
        print("📤 Upload des artefacts dans MLflow...")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(local_weights), "model")
            print("   ✅ best.pt uploadé")
            
            # Upload aussi args.yaml et results.csv
            args_file = local_weights.parent.parent / "args.yaml"
            if args_file.exists():
                mlflow.log_artifact(str(args_file), "config")
                print("   ✅ args.yaml uploadé")
            
            results_file = local_weights.parent.parent / "results.csv"
            if results_file.exists():
                mlflow.log_artifact(str(results_file), "results")
                print("   ✅ results.csv uploadé")
        
        print()
        print("✅ Artefacts uploadés avec succès!")
        print()
        
        # Enregistrer dans Model Registry
        print(f"🏷️ Enregistrement dans Model Registry '{MODEL_NAME}'...")
        
        model_uri = f"runs:/{run_id}/model/best.pt"
        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        
        print()
        print("=" * 60)
        print("🎉 SUCCÈS!")
        print("=" * 60)
        print(f"📦 Modèle: {MODEL_NAME}")
        print(f"🔢 Version: {result.version}")
        print(f"📊 mAP@50: {mAP50:.4f}")
        print(f"🔗 Run: {run_name}")
        print()
        print(f"🌐 Voir dans MLflow UI:")
        print(f"   {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")
        print()
        print("📝 Prochaines étapes:")
        print("   1. Ouvrir MLflow UI → Models → yolo_person_detector")
        print("   2. Transition vers stage 'Staging'")
        print("   3. Ajouter une description")
        print("   4. Capturer screenshots")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        print()
        print("💡 SOLUTION MANUELLE:")
        print(f"   1. Ouvrir: {MLFLOW_TRACKING_URI}")
        print(f"   2. Run: {run_name}")
        print("   3. Artifacts → weights/best.pt → Register Model")
        print(f"   4. Nom: {MODEL_NAME}, Stage: Staging")

if __name__ == "__main__":
    upload_and_register()

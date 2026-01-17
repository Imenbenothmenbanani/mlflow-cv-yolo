"""
Script pour enregistrer le meilleur modèle dans MLflow Model Registry
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def register_best_model(experiment_name="cv_yolo_tiny", model_name="yolo_person_detector"):
    """
    Trouve le meilleur run et enregistre le modèle dans le Registry
    """
    print(f"🔍 Recherche du meilleur modèle dans '{experiment_name}'...")
    
    try:
        # Récupérer l'expérience
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Expérience '{experiment_name}' non trouvée!")
            return
        
        # Trouver le meilleur run basé sur mAP@50
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.mAP50 DESC"],
            max_results=1
        )
        
        if runs.empty:
            print("❌ Aucun run trouvé!")
            return
        
        best_run = runs.iloc[0]
        run_id = best_run['run_id']
        
        print(f"\n✅ Meilleur run trouvé:")
        print(f"   Run ID: {run_id}")
        print(f"   mAP@50: {best_run.get('metrics.mAP50', 'N/A')}")
        print(f"   Image Size: {best_run.get('params.imgsz', 'N/A')}")
        print(f"   Learning Rate: {best_run.get('params.lr0', 'N/A')}")
        
        # Vérifier si des artefacts existent
        client = MlflowClient()
        run_data = client.get_run(run_id)
        artifact_uri = run_data.info.artifact_uri
        
        print(f"\n📦 Artifacts URI: {artifact_uri}")
        
        # Note: L'enregistrement manuel est préféré via l'UI MLflow
        # car les artefacts YOLO ne sont pas au format MLflow standard
        
        print("\n" + "=" * 60)
        print("ℹ️  ENREGISTREMENT MANUEL RECOMMANDÉ:")
        print("=" * 60)
        print("\n1. Ouvrir MLflow UI: http://localhost:5000")
        print(f"2. Aller dans l'expérience '{experiment_name}'")
        print(f"3. Cliquer sur le run: {run_id[:8]}...")
        print("4. Dans l'onglet 'Artifacts', sélectionner 'weights/best.pt'")
        print("5. Cliquer sur 'Register Model'")
        print(f"6. Nom du modèle: {model_name}")
        print("7. Choisir le stage: 'Staging' ou 'Production'")
        print("\n✅ Prendre une CAPTURE D'ÉCRAN de la confirmation!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    register_best_model()

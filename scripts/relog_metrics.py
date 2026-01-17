"""
Script pour ré-logger les métriques des runs existants dans MLflow
Les runs ont été exécutés mais les métriques n'ont pas été loggées
"""
import os
import mlflow
import pandas as pd
from pathlib import Path

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def relog_metrics_for_run(run_id, results_csv_path):
    """Re-log les métriques d'un run à partir de son results.csv"""
    
    if not Path(results_csv_path).exists():
        print(f"  ❌ Fichier non trouvé: {results_csv_path}")
        return False
    
    try:
        # Lire le CSV
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()  # Nettoyer les espaces
        
        # Prendre la dernière ligne (fin de l'entraînement)
        last = df.iloc[-1].to_dict()
        
        # Mapping des colonnes possibles
        metrics_to_log = {}
        
        # Essayer différentes variantes de noms de colonnes
        candidates = {
            "precision": ["metrics/precision(B)", "metrics/precision"],
            "recall": ["metrics/recall(B)", "metrics/recall"],
            "mAP50": ["metrics/mAP50(B)", "metrics/mAP50"],
            "mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(M)"],
        }
        
        for metric_name, possible_cols in candidates.items():
            for col in possible_cols:
                if col in last:
                    try:
                        value = float(last[col])
                        metrics_to_log[metric_name] = value
                        break
                    except (ValueError, TypeError):
                        pass
        
        # Logger les métriques dans MLflow
        if metrics_to_log:
            client = mlflow.tracking.MlflowClient()
            for metric_name, value in metrics_to_log.items():
                try:
                    client.log_metric(run_id, metric_name, value)
                except Exception as e:
                    print(f"    ⚠️ Erreur lors du log de {metric_name}: {e}")
            
            print(f"  ✅ Métriques loggées: {list(metrics_to_log.keys())}")
            print(f"     mAP@50: {metrics_to_log.get('mAP50', 'N/A'):.4f}")
            return True
        else:
            print(f"  ⚠️ Aucune métrique trouvée dans le CSV")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    print("🔧 Re-logging des métriques des runs existants")
    print("=" * 60)
    
    try:
        # Récupérer tous les runs de l'expérience
        experiment = mlflow.get_experiment_by_name("cv_yolo_tiny")
        if not experiment:
            print("❌ Expérience 'cv_yolo_tiny' non trouvée!")
            return
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            print("❌ Aucun run FINISHED trouvé")
            return
        
        print(f"✅ {len(runs)} run(s) FINISHED trouvé(s)\n")
        
        # Parcourir les runs et chercher leurs results.csv
        success_count = 0
        
        for idx, run in runs.iterrows():
            run_id = run['run_id']
            run_name = run.get('tags.mlflow.runName', 'Unknown')
            
            print(f"\n📊 Run {idx+1}/{len(runs)}: {run_name}")
            print(f"   ID: {run_id[:8]}...")
            
            # Chercher le dossier de résultats local
            # Le nom du run correspond au nom du dossier
            possible_paths = [
                Path(f"runs/{run_name}/results.csv"),
                Path(f"runs/detect/{run_name}/results.csv"),
            ]
            
            # Essayer aussi de chercher dans tous les sous-dossiers de runs/
            runs_dir = Path("runs")
            if runs_dir.exists():
                for subdir in runs_dir.iterdir():
                    if subdir.is_dir() and run_name in subdir.name:
                        possible_paths.append(subdir / "results.csv")
            
            results_found = False
            for csv_path in possible_paths:
                if csv_path.exists():
                    print(f"   📁 Trouvé: {csv_path}")
                    if relog_metrics_for_run(run_id, csv_path):
                        success_count += 1
                        results_found = True
                    break
            
            if not results_found:
                print(f"   ⚠️ Aucun results.csv trouvé pour ce run")
        
        print("\n" + "=" * 60)
        print(f"✅ Terminé! {success_count}/{len(runs)} runs mis à jour")
        print(f"\n🌐 Voir les résultats: {MLFLOW_TRACKING_URI}")
        print("\n💡 Relancez maintenant: python scripts/analyze_runs.py")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

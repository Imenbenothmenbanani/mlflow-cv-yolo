"""
Script d'analyse des runs MLflow pour le TP4
Génère un rapport de comparaison et aide à remplir le template de décision
"""
import os
import mlflow
import pandas as pd
from pathlib import Path

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def analyze_experiment(experiment_name="cv_yolo_tiny"):
    """Analyse tous les runs d'une expérience"""
    
    print(f"🔍 Analyse de l'expérience: {experiment_name}")
    print("=" * 60)
    
    try:
        # Récupérer l'expérience
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Expérience '{experiment_name}' non trouvée!")
            print("\n💡 Assurez-vous que:")
            print("   1. MLflow est démarré (docker compose up -d)")
            print("   2. Des runs ont été exécutés")
            return
        
        # Récupérer tous les runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            print(f"❌ Aucun run trouvé dans l'expérience '{experiment_name}'")
            return
        
        print(f"✅ {len(runs)} run(s) trouvé(s)\n")
        
        # Colonnes importantes
        metrics_cols = [col for col in runs.columns if col.startswith('metrics.')]
        params_cols = [col for col in runs.columns if col.startswith('params.')]
        
        # Afficher le résumé
        print("\n📊 RÉSUMÉ DES RUNS:")
        print("-" * 60)
        
        summary_cols = ['run_id', 'start_time', 'status']
        display_params = ['params.imgsz', 'params.lr0', 'params.seed', 'params.epochs']
        display_metrics = ['metrics.mAP50', 'metrics.mAP50-95', 'metrics.precision', 'metrics.recall']
        
        # Filtrer les colonnes qui existent
        summary_cols += [col for col in display_params if col in runs.columns]
        summary_cols += [col for col in display_metrics if col in runs.columns]
        
        summary_df = runs[summary_cols].copy()
        
        # Renommer pour plus de clarté
        rename_dict = {
            'params.imgsz': 'img_size',
            'params.lr0': 'lr',
            'params.seed': 'seed',
            'params.epochs': 'epochs',
            'metrics.mAP50': 'mAP@50',
            'metrics.mAP50-95': 'mAP@50-95',
            'metrics.precision': 'precision',
            'metrics.recall': 'recall'
        }
        summary_df = summary_df.rename(columns=rename_dict)
        
        # Afficher avec pandas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 20)
        
        print(summary_df.to_string(index=False))
        
        # Statistiques
        print("\n📈 STATISTIQUES DES MÉTRIQUES:")
        print("-" * 60)
        
        metric_stats = {}
        for col in display_metrics:
            if col in runs.columns:
                metric_name = rename_dict.get(col, col)
                values = runs[col].dropna()
                if not values.empty:
                    metric_stats[metric_name] = {
                        'min': values.min(),
                        'max': values.max(),
                        'mean': values.mean(),
                        'std': values.std()
                    }
        
        stats_df = pd.DataFrame(metric_stats).T
        print(stats_df.round(4))
        
        # Trouver le meilleur run (basé sur mAP@50)
        print("\n🏆 MEILLEUR RUN (basé sur mAP@50):")
        print("-" * 60)
        
        if 'metrics.mAP50' in runs.columns:
            best_idx = runs['metrics.mAP50'].idxmax()
            best_run = runs.loc[best_idx]
            
            print(f"Run ID: {best_run['run_id'][:8]}...")
            print(f"Status: {best_run['status']}")
            if 'params.imgsz' in runs.columns:
                print(f"Image Size: {best_run['params.imgsz']}")
            if 'params.lr0' in runs.columns:
                print(f"Learning Rate: {best_run['params.lr0']}")
            if 'params.seed' in runs.columns:
                print(f"Seed: {best_run['params.seed']}")
            if 'metrics.mAP50' in runs.columns:
                print(f"mAP@50: {best_run['metrics.mAP50']:.4f}")
            if 'metrics.mAP50-95' in runs.columns:
                print(f"mAP@50-95: {best_run['metrics.mAP50-95']:.4f}")
            if 'metrics.precision' in runs.columns:
                print(f"Precision: {best_run['metrics.precision']:.4f}")
            if 'metrics.recall' in runs.columns:
                print(f"Recall: {best_run['metrics.recall']:.4f}")
        
        # Sauvegarder le rapport
        report_path = Path("reports") / "runs_analysis.csv"
        report_path.parent.mkdir(exist_ok=True)
        summary_df.to_csv(report_path, index=False)
        print(f"\n💾 Rapport sauvegardé: {report_path}")
        
        print("\n" + "=" * 60)
        print("✅ Analyse terminée!")
        print(f"\n🌐 Voir dans MLflow UI: {MLFLOW_TRACKING_URI}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_experiment()

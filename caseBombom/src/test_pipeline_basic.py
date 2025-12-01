"""Teste básico do pipeline: executa e checa artefatos mínimos na pasta final_delivery.
Execute com: python caseBombom/src/test_pipeline_basic.py
"""
from pathlib import Path # Para manipulação de caminhos de arquivos
import sys
import subprocess
import json

BASE_DIR = Path(__file__).resolve().parent.parent
PIPE = BASE_DIR / 'src' / 'pipeline_full.py'
FINAL_DIR = BASE_DIR / 'outputs' / 'final_delivery'

def run_pipeline():
    print('Executando pipeline para teste...')
    res = subprocess.run([sys.executable, str(PIPE)], capture_output=True, text=True)
    if res.returncode != 0:
        print('STDOUT:\n', res.stdout)
        print('STDERR:\n', res.stderr)
        raise SystemExit('Falha na execução do pipeline.')

def check_artifacts(root: Path):
    required = [
        root / 'metrics' / 'model_metrics.csv',
        root / 'metrics' / 'cv_metrics.csv',
        root / 'metrics' / 'prescriptive_optimum.csv',
        root / 'tables' / 'custo_grade.csv',
        root / 'tables' / 'diagnostic_correlations.csv',
        root / 'tables' / 'descriptive.json',
        root / 'tables' / 'feature_importances.csv',
        root / 'figures' / 'custo_risco.png',
        root / 'figures' / 'curva_predicao.png',
        root / 'models' / 'random_forest.pkl',
        root / 'models' / 'linear_regression.pkl',
        root / 'summary.json',
        root / 'summary.txt',
        root / 'run_config.json'
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f'Arquivos faltantes: {missing}')
    print('Artefatos principais OK.')
    with open(root / 'summary.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    print('Resumo:', {k: summary.get(k) for k in ['qtd_opt','best_model']})
    # Plano por VAR_1 é opcional
    plan_path = root / 'tables' / 'production_plan_var1.csv'
    if plan_path.exists():
        import pandas as pd
        plan = pd.read_csv(plan_path)
        print('Plano VAR_1 (primeiras linhas):')
        print(plan.head())

if __name__ == '__main__':
    run_pipeline()
    if not FINAL_DIR.exists():
        raise SystemExit('Pasta final_delivery não foi criada.')
    check_artifacts(FINAL_DIR)
    print('Teste básico concluído com sucesso.')

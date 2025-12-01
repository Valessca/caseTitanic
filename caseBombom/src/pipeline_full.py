from __future__ import annotations
"""Pipeline completo para o Case Bombom.

Etapas:
1. Carregamento e validação dos dados
2. Análise Descritiva (estatísticas, correlação base)
3. Análise Diagnóstica (correlações Pearson/Spearman ordenadas)
4. Treino de modelos (Linear vs RandomForest) + métricas
5. Simulação e Otimização Prescritiva (QTD_CHOC ótimo dado risco)
6. Persistência organizada de artefatos (metrics, figures, tables)
7. Geração de resumo executivo (summary.txt / summary.json)

Estrutura de saída:
outputs/ runs/<timestamp>/
    metrics/ (model_metrics.csv, validation_metrics.csv)
    tables/ (descriptive.json, diagnostic.csv, production_plan.csv, custo_grade.csv)
    figures/ (hist_peso.png, importances.png, custo_risco.png, curva_predicao.png)

Uso:
python -m caseBombom.src.pipeline_full
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib  # para serializar modelos
from sklearn.inspection import permutation_importance  # novo

# --- Configurações globais ---
np.random.seed(42)
plt.style.use('seaborn-v0_8')
BASE_DIR = Path(__file__).resolve().parent.parent  # caseBombom/
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'registros_train.csv'

# --- Funções de dados ---
EXPECTED_COLS = {'QTD_CHOC','VAR_1','VAR_2','PESO_BOMBOM'}

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Dados não encontrados em {path}')
    df = pd.read_csv(path, sep=';', decimal=',')
    df.columns = [c.strip() for c in df.columns]
    miss = EXPECTED_COLS - set(df.columns)
    if miss:
        raise ValueError(f'Colunas faltantes: {miss}')
    for col in ['QTD_CHOC','VAR_1','PESO_BOMBOM']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['QTD_CHOC','VAR_1','PESO_BOMBOM']].isna().any().any():
        raise ValueError('Falha conversão numérica — verifique separador decimal.')
    df['VAR_2'] = df['VAR_2'].astype('category')
    return df

# --- Análise Descritiva ---

def descriptive_stats(df: pd.DataFrame) -> Dict[str, any]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T
    desc['cv'] = desc['std'] / desc['mean']
    desc['iqr'] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    return {
        'shape': df.shape,
        'numeric_cols': num_cols,
        'categorical_cols': cat_cols,
        'describe': desc.reset_index().rename(columns={'index':'col'}).to_dict(orient='records')
    }

# --- Análise Diagnóstica ---

def diagnostic_correlations(df: pd.DataFrame, target: str = 'PESO_BOMBOM') -> pd.DataFrame:
    results = []
    for col in df.columns:
        if col == target: continue
        x = pd.to_numeric(df[col], errors='coerce')
        y = pd.to_numeric(df[target], errors='coerce')
        mask = x.notna() & y.notna()
        if mask.sum() < 5: continue
        pr, pp = stats.pearsonr(x[mask], y[mask])
        sr, sp = stats.spearmanr(x[mask], y[mask])
        results.append({'feature': col, 'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp})
    diag = pd.DataFrame(results)
    if not diag.empty:
        diag = diag.sort_values(by='spearman_r', key=lambda s: s.abs(), ascending=False)
    return diag

# --- Modelagem ---

def prepare_model_matrix(df: pd.DataFrame, target: str = 'PESO_BOMBOM'):
    df_model = pd.get_dummies(df, columns=['VAR_2'], drop_first=True)
    X = df_model.drop(target, axis=1)
    y = df_model[target]
    return X, y

@dataclass
class ModelResults:
    metrics: pd.DataFrame
    lin_model: LinearRegression
    rf_model: RandomForestRegressor
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    pred_lin: np.ndarray
    pred_rf: np.ndarray


def train_models(X: pd.DataFrame, y: pd.Series) -> ModelResults:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    # Linear
    lin = LinearRegression().fit(X_train, y_train)
    pred_lin = lin.predict(X_val)
    # RandomForest
    rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(X_train, y_train)
    pred_rf = rf.predict(X_val)
    metrics_lin = {
        'model':'Linear',
        'RMSE': mean_squared_error(y_val, pred_lin) ** 0.5,
        'MAE': mean_absolute_error(y_val, pred_lin),
        'R2': r2_score(y_val, pred_lin)
    }
    metrics_rf = {
        'model':'RandomForest',
        'RMSE': mean_squared_error(y_val, pred_rf) ** 0.5,
        'MAE': mean_absolute_error(y_val, pred_rf),
        'R2': r2_score(y_val, pred_rf)
    }
    metrics_df = pd.DataFrame([metrics_lin, metrics_rf])
    return ModelResults(metrics_df, lin, rf, X_train, X_val, y_train, y_val, pred_lin, pred_rf)

# --- Função de custo + simulação ---

def cost_function(p: float, low=9.0, target=10.0, penalty_nonconf=100.0, shape_below=4, coeff_above=1.0) -> float:
    if p < low:
        return penalty_nonconf + (low - p) * penalty_nonconf/2
    elif p < target:
        return (target - p)**shape_below * (penalty_nonconf/10)
    else:
        return (p - target)**2 * coeff_above


def simulate_qtd_curve(rf: RandomForestRegressor, X_template: pd.Series, qtd_values: np.ndarray, residuals: np.ndarray, n_sim: int = 1500) -> Dict[str, np.ndarray]:
    sims = {}
    for q in qtd_values:
        row = X_template.copy()
        if 'QTD_CHOC' in row.index:
            row['QTD_CHOC'] = q
        pred = rf.predict(pd.DataFrame([row]))[0]
        sampled = np.random.choice(residuals, size=n_sim, replace=True)
        sims[q] = pred + sampled
    return sims


def prescriptive_optimization(rf: RandomForestRegressor, X: pd.DataFrame, y_val: pd.Series, pred_rf: np.ndarray, max_prob_nonconf: float = 0.05) -> Tuple[float, pd.DataFrame]:
    residuals = y_val.values - pred_rf
    template = X.median()  # usar medianas como referência
    qtd_min, qtd_max = X['QTD_CHOC'].min(), X['QTD_CHOC'].max()
    grid = np.linspace(qtd_min, qtd_max, 60)
    simulations = simulate_qtd_curve(rf, template, grid, residuals, n_sim=1200)
    expected_cost = []
    prob_nonconf = []
    for q in grid:
        arr = simulations[q]
        costs = [cost_function(v) for v in arr]
        expected_cost.append(np.mean(costs))
        prob_nonconf.append(np.mean(arr < 9.0))
    expected_cost = np.array(expected_cost)
    prob_nonconf = np.array(prob_nonconf)
    mask = prob_nonconf <= max_prob_nonconf
    if mask.any():
        idx = np.argmin(expected_cost[mask])
        idx_global = np.where(mask)[0][idx]
    else:
        idx_global = np.argmin(prob_nonconf)  # fallback
    qtd_opt = float(grid[idx_global])
    table = pd.DataFrame({'QTD_CHOC': grid, 'expected_cost': expected_cost, 'prob_nonconf': prob_nonconf})
    return qtd_opt, table

# --- Persistência ---

def make_run_dirs(base_out: Path) -> Dict[str, Path]:
    from datetime import datetime
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M')
    run_root = base_out / 'runs' / run_id
    dirs = {
        'root': run_root,
        'metrics': run_root / 'metrics',
        'tables': run_root / 'tables',
        'figures': run_root / 'figures'
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

# --- Geração de figuras ---

def plot_weight_distribution(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    sns.histplot(df['PESO_BOMBOM'], kde=True, ax=ax[0], color='steelblue')
    ax[0].axvline(10, color='red', linestyle='--', label='Alvo 10g')
    ax[0].axvline(9, color='orange', linestyle='--', label='Limite 9g')
    ax[0].legend(); ax[0].set_title('Distribuição Peso')
    sns.boxplot(y=df['PESO_BOMBOM'], ax=ax[1], color='lightblue')
    ax[1].set_title('Boxplot Peso')
    fig.tight_layout()
    fig.savefig(out / 'hist_peso.png', dpi=120)
    plt.close(fig)


def plot_feature_importances(rf: RandomForestRegressor, X: pd.DataFrame, out: Path):
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(x=imp.values, y=imp.index, color='tab:blue')
    for i,(v,name) in enumerate(zip(imp.values, imp.index)):
        plt.text(v+0.002, i, f"{v*100:.1f}%", va='center', fontsize=8)
    plt.title('Importância das Features (Impureza)')
    plt.tight_layout()
    plt.savefig(out / 'importances.png', dpi=120)
    plt.close()
    return imp

# Novo: importâncias por permutação

def plot_permutation_importances(rf: RandomForestRegressor, X: pd.DataFrame, y: pd.Series, out: Path):
    result = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=1)
    imp_perm = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=imp_perm.values, y=imp_perm.index, color='tab:green', ax=ax)
    for i,(v,name) in enumerate(zip(imp_perm.values, imp_perm.index)):
        ax.text(v+0.0005, i, f"{v:.3f}", va='center', fontsize=8)
    ax.set_title('Importância das Features (Permutação)')
    ax.set_xlabel('Queda média de performance (Δ)')
    fig.tight_layout()
    fig.savefig(out / 'importances_permutation.png', dpi=120)
    plt.close(fig)
    # salvar tabela
    imp_perm.to_frame('perm_importance').reset_index().rename(columns={'index':'feature'}).to_csv(out / 'feature_importances_permutation.csv', index=False)
    return imp_perm

# Funções complementares de visualização (repostas para custo e curva de predição)

def plot_cost_risk(table: pd.DataFrame, qtd_opt: float, out: Path):
    fig, ax1 = plt.subplots(figsize=(9,5))
    ax1.plot(table['QTD_CHOC'], table['expected_cost'], color='tab:blue', label='Custo Esperado')
    ax1.axvline(qtd_opt, color='green', linestyle='--', label=f'Ótimo {qtd_opt:.2f}')
    ax1.set_xlabel('QTD_CHOC'); ax1.set_ylabel('Custo', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(table['QTD_CHOC'], table['prob_nonconf'], color='tab:red', label='Prob <9g')
    ax2.axhline(0.05, color='red', linestyle=':', label='Limite 5%')
    ax2.set_ylabel('Prob Não Conformidade', color='tab:red')
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='upper right')
    plt.title('Custo vs Risco')
    fig.tight_layout(); fig.savefig(out / 'custo_risco.png', dpi=120)
    plt.close(fig)

def plot_prediction_curve(rf: RandomForestRegressor, X: pd.DataFrame, out: Path):
    template = X.median()
    grid = np.linspace(X['QTD_CHOC'].min(), X['QTD_CHOC'].max(), 60)
    preds = []
    for q in grid:
        row = template.copy(); row['QTD_CHOC'] = q
        preds.append(rf.predict(pd.DataFrame([row]))[0])
    preds = np.array(preds)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(grid, preds, label='Predição Média', color='navy')
    ax.axhline(10, color='black', linestyle='--', label='Alvo 10g')
    ax.set_xlabel('QTD_CHOC'); ax.set_ylabel('PESO_BOMBOM (pred)')
    ax.set_title('Curva de Predição vs QTD_CHOC')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'curva_predicao.png', dpi=120)
    plt.close(fig)

# --- Salvamento de artefatos complementares ---

def save_feature_importances_table(imp: 'pd.Series', out: Path):
    imp_df = imp.reset_index().rename(columns={'index': 'feature', 0: 'importance'})
    # Caso pandas renomeie colunas diferentemente, garantir nome correto
    if 'importance' not in imp_df.columns:
        # busca coluna numérica
        for c in imp_df.columns:
            if c != 'feature':
                imp_df = imp_df.rename(columns={c: 'importance'})
                break
    imp_df.to_csv(out / 'feature_importances.csv', index=False)
    return imp_df


def write_config(out: Path, config: dict):
    with open(out / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def write_summary(out: Path, qtd_opt: float, metrics_df: pd.DataFrame, cv_r2: np.ndarray, cv_rmse: np.ndarray, cost_table: pd.DataFrame, imp_df: pd.DataFrame):
    summary_lines = []
    summary_lines.append('# Resumo Executivo da Execução\n')
    summary_lines.append(f"QTD_CHOC ótimo estimado (risco <=5% não conformes): {qtd_opt:.2f}\n")
    best_row = metrics_df.sort_values(by='R2', ascending=False).iloc[0]
    summary_lines.append('Melhor modelo (maior R2 validação): ' + best_row['model'] + '\n')
    summary_lines.append(f"Métricas {best_row['model']}: R2={best_row['R2']:.3f} RMSE={best_row['RMSE']:.3f} MAE={best_row['MAE']:.3f}\n")
    summary_lines.append(f"Cross-Validation RF: R2 médio={cv_r2.mean():.3f} (±{cv_r2.std():.3f}); RMSE médio={cv_rmse.mean():.3f}\n")
    summary_lines.append('Top 5 importâncias de features (RF):\n')
    for _, r in imp_df.sort_values(by='importance', ascending=False).head(5).iterrows():
        summary_lines.append(f"  - {r['feature']}: {r['importance']:.4f}\n")
    # Ponto de menor custo dentro da máscara de risco
    mask = cost_table['prob_nonconf'] <= 0.05
    if mask.any():
        ct_sub = cost_table[mask]
        idx_min = ct_sub['expected_cost'].idxmin()
        row = cost_table.loc[idx_min]
        summary_lines.append(f"Menor custo esperado dentro da restrição: QTD_CHOC={row['QTD_CHOC']:.2f} custo={row['expected_cost']:.3f} prob_nonconf={row['prob_nonconf']:.3f}\n")
    else:
        summary_lines.append('Nenhum ponto atende restrição de risco — escolhido menor prob_nonconf disponível.\n')
    summary_text = ''.join(summary_lines)
    with open(out / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    # JSON
    summary_json = {
        'qtd_opt': qtd_opt,
        'best_model': best_row['model'],
        'best_model_metrics': {k: float(best_row[k]) if k in ['RMSE','MAE','R2'] else best_row[k] for k in ['RMSE','MAE','R2']},
        'cv_rf': {
            'r2_mean': float(cv_r2.mean()), 'r2_std': float(cv_r2.std()),
            'rmse_mean': float(cv_rmse.mean()), 'rmse_std': float(cv_rmse.std())
        },
        'importances_top5': [
            {'feature': r['feature'], 'importance': float(r['importance'])}
            for _, r in imp_df.sort_values(by='importance', ascending=False).head(5).iterrows()
        ]
    }
    with open(out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

# --- Execução Principal ---

def main():
    print('Iniciando pipeline completo...')
    df = load_data(DATA_PATH)
    print(f'Dados carregados: shape={df.shape}')

    # 1. Descritiva
    desc = descriptive_stats(df)
    # 2. Diagnóstica
    diag = diagnostic_correlations(df)
    # 3. Modelagem
    X, y = prepare_model_matrix(df)
    results = train_models(X, y)
    # 3b. Cross-validation RF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(results.rf_model, X, y, cv=kf, scoring='r2')
    cv_rmse = -cross_val_score(results.rf_model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    # 4. Prescritiva
    qtd_opt, cost_table = prescriptive_optimization(results.rf_model, X, results.y_val, results.pred_rf, max_prob_nonconf=0.05)

    # Pastas
    out_dirs = make_run_dirs(BASE_DIR / 'outputs')

    # Config usada
    run_config = {
        'data_path': str(DATA_PATH),
        'random_seed': 42,
        'model': 'RandomForest(n_estimators=500)',
        'prescriptive': {
            'risk_threshold_nonconf': 0.05,
            'grid_points': 60,
            'cost_function_params': {
                'low': 9.0,
                'target': 10.0,
                'penalty_nonconf': 100.0,
                'shape_below': 4,
                'coeff_above': 1.0
            }
        }
    }
    write_config(out_dirs['root'], run_config)

    # Salvar métricas
    results.metrics.to_csv(out_dirs['metrics'] / 'model_metrics.csv', index=False)
    pd.DataFrame({'CV_R2': cv_r2, 'CV_RMSE': cv_rmse}).to_csv(out_dirs['metrics'] / 'cv_metrics.csv', index=False)
    pd.DataFrame([{'qtd_opt': qtd_opt}]).to_csv(out_dirs['metrics'] / 'prescriptive_optimum.csv', index=False)

    # Salvar tabelas
    with open(out_dirs['tables'] / 'descriptive.json', 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)
    diag.to_csv(out_dirs['tables'] / 'diagnostic_correlations.csv', index=False)
    cost_table.to_csv(out_dirs['tables'] / 'custo_grade.csv', index=False)

    # Figuras + importâncias
    plot_weight_distribution(df, out_dirs['figures'])
    imp = plot_feature_importances(results.rf_model, X, out_dirs['figures'])
    imp_df = save_feature_importances_table(imp, out_dirs['tables'])
    # Permutation importances (mais robusto contra vieses de variável contínua)
    imp_perm = plot_permutation_importances(results.rf_model, X, y, out_dirs['figures'])
    # Copiar tabela para tables/
    imp_perm.to_frame('perm_importance').reset_index().rename(columns={'index':'feature'}).to_csv(out_dirs['tables'] / 'feature_importances_permutation.csv', index=False)
    plot_cost_risk(cost_table, qtd_opt, out_dirs['figures'])
    plot_prediction_curve(results.rf_model, X, out_dirs['figures'])

    # Summary (incluir menção ao plano por VAR_1 se gerado)
    write_summary(out_dirs['root'], qtd_opt, results.metrics, cv_r2, cv_rmse, cost_table, imp_df)
    # Apêndice rápido de top 3 permutação
    try:
        top3 = imp_perm.head(3)
        with open(out_dirs['root'] / 'summary.txt', 'a', encoding='utf-8') as f:
            f.write('\nTop 3 importâncias (Permutação):\n')
            for feat, val in top3.items():
                f.write(f"  - {feat}: Δ={val:.4f}\n")
    except Exception:
        pass
    if not plan_var1.empty:
        plan_var1.head().to_csv(out_dirs['metrics'] / 'production_plan_var1_head.csv', index=False)

    # Serializar modelos
    joblib.dump(results.rf_model, models_dir / 'random_forest.pkl')
    joblib.dump(results.lin_model, models_dir / 'linear_regression.pkl')
    np.save(models_dir / 'residuals_val.npy', results.y_val.values - results.pred_rf)
    med_dict = {c: float(X[c].median()) for c in X.columns}
    with open(models_dir / 'features_median.json', 'w', encoding='utf-8') as f:
        json.dump(med_dict, f, ensure_ascii=False, indent=2)

    print('Pipeline concluído (final_delivery).')
    print(f"QTD_CHOC ótimo (risco <=5%): {qtd_opt:.2f}")
    if not plan_var1.empty:
        print('Plano por VAR_1 salvo em tables/production_plan_var1.csv')
    print('Diretório de saída:', out_dirs['root'])

if __name__ == '__main__':
    main()

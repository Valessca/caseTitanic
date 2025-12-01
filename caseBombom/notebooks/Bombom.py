import warnings, os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

np.random.seed(42)
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', 50)

# 0. Carregamento dos dados
base_dir = Path(__file__).resolve().parent.parent  # caseBombom
data_path = base_dir / 'data' / 'raw' / 'registros_train.csv'
df = pd.read_csv(data_path, sep=';', decimal=',')
print('Shape:', df.shape)
print(df.head())

# 1. Tipos e valores ausentes
print(df.dtypes)
missing = df.isna().sum().to_frame('missing')
missing['missing_pct'] = 100*missing['missing']/len(df)
print(missing)

# 2. Análise Descritiva
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print('Numéricas:', numeric_cols)
print('Categóricas:', cat_cols)

desc = df[numeric_cols].describe().T
desc['cv'] = desc['std'] / desc['mean']
desc['iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
print(desc)

# Distribuição do alvo
fig, ax = plt.subplots(1,2, figsize=(10,4))
sns.histplot(df['PESO_BOMBOM'], kde=True, ax=ax[0], color='steelblue')
ax[0].axvline(10, color='red', linestyle='--', label='Peso alvo 10g')
ax[0].axvline(9, color='orange', linestyle='--', label='Limite 9g')
ax[0].set_title('Distribuição PESO_BOMBOM')
ax[0].legend()
sns.boxplot(y=df['PESO_BOMBOM'], ax=ax[1], color='lightblue')
ax[1].set_title('Boxplot PESO_BOMBOM')
plt.tight_layout(); plt.show()

# Relações principais
sns.pairplot(df[['QTD_CHOC','VAR_1','PESO_BOMBOM']], corner=True, diag_kind='kde')
plt.suptitle('Scatter / KDE principais variáveis', y=1.02)
plt.show()

# Heatmap correlações
corr = df[numeric_cols].corr() # Correlação Pearson
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlação Pearson (numéricas)')
plt.show()
print(corr['PESO_BOMBOM'].drop('PESO_BOMBOM').sort_values(key=lambda s: s.abs(), ascending=False))

# Outliers simples
outlier_info = {}
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    mask = (df[col] < low) | (df[col] > high)
    outlier_info[col] = {'low': float(low), 'high': float(high), 'outlier_pct': 100*mask.mean()}
print(pd.DataFrame(outlier_info).T)

# 3. Análise Diagnóstica

def compute_correlations(df, target):
    results = []
    for col in df.columns:
        if col == target: continue
        if df[col].dtype.kind in 'bifc':
            x = df[col]; y = df[target]
            pr, pp = stats.pearsonr(x, y); sr, sp = stats.spearmanr(x, y)
            results.append({'feature': col, 'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp})
        else:
            codes = pd.Categorical(df[col]).codes
            pr, pp = stats.pearsonr(codes, df[target]); sr, sp = stats.spearmanr(codes, df[target])
            results.append({'feature': col, 'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp})
    out = pd.DataFrame(results)
    return out.sort_values(by='spearman_r', key=lambda s: s.abs(), ascending=False)

print(compute_correlations(df, 'PESO_BOMBOM'))

# Correlação parcial controlando QTD_CHOC

def partial_corr(x, y, control):
    x_res = x - np.poly1d(np.polyfit(control, x, 1))(control)
    y_res = y - np.poly1d(np.polyfit(control, y, 1))(control)
    r, p = stats.pearsonr(x_res, y_res)
    return r, p

pc_results = []
control = df['QTD_CHOC']
r, p = partial_corr(df['VAR_1'], df['PESO_BOMBOM'], control)
pc_results.append({'feature': 'VAR_1', 'partial_r': r, 'p_value': p})
for lvl in df['VAR_2'].unique():
    mask = (df['VAR_2'] == lvl).astype(int)
    r, p = partial_corr(mask, df['PESO_BOMBOM'], control)
    pc_results.append({'feature': f'VAR_2_{lvl}', 'partial_r': r, 'p_value': p})
print(pd.DataFrame(pc_results).sort_values(by='partial_r', key=lambda s: s.abs(), ascending=False))

# Regressão linear explicativa completa
_df_lin = pd.get_dummies(df.copy(), columns=['VAR_2'], drop_first=True)
X_lin = _df_lin.drop('PESO_BOMBOM', axis=1); y_lin = _df_lin['PESO_BOMBOM']
lin_model = LinearRegression().fit(X_lin, y_lin)
coefs = pd.Series(lin_model.coef_, index=X_lin.columns).sort_values(key=lambda s: s.abs(), ascending=False)
print('Coeficientes (não padronizados):')
print(coefs)

# Resíduos modelo linear
res_lin = y_lin - lin_model.predict(X_lin)
fig, ax = plt.subplots(1,2, figsize=(10,4))
sns.scatterplot(x=lin_model.predict(X_lin), y=res_lin, ax=ax[0])
ax[0].axhline(0, color='red', linestyle='--'); ax[0].set_title('Resíduos vs Predito')
stats.probplot(res_lin, dist='norm', plot=ax[1]); ax[1].set_title('QQ Plot Resíduos')
plt.tight_layout(); plt.show()

# 4. Análise Preditiva
# Preparação para modelagem
_df_model = pd.get_dummies(df.copy(), columns=['VAR_2'], drop_first=True)
X = _df_model.drop('PESO_BOMBOM', axis=1); y = _df_model['PESO_BOMBOM']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
print('Shapes treino/val:', X_train.shape, X_val.shape)

# Modelo 1: Linear
lin_reg = LinearRegression().fit(X_train, y_train)
pred_lin = lin_reg.predict(X_val)
rmse_lin = np.sqrt(mean_squared_error(y_val, pred_lin))
metrics_lin = {'model': 'Linear', 'RMSE': rmse_lin, 'MAE': mean_absolute_error(y_val, pred_lin), 'R2': r2_score(y_val, pred_lin)}
print('Metrics Linear:', metrics_lin)

# Modelo 2: Random Forest
rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(X_train, y_train)
pred_rf = rf.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))
metrics_rf = {'model': 'RandomForest', 'RMSE': rmse_rf, 'MAE': mean_absolute_error(y_val, pred_rf), 'R2': r2_score(y_val, pred_rf)}
print('Metrics RF:', metrics_rf)

metrics_df = pd.DataFrame([metrics_lin, metrics_rf])
print(metrics_df)

# Cross-Validation ajustada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring='r2')
cv_mse = -cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(cv_mse)
print(pd.DataFrame({'CV_R2': cv_r2, 'CV_RMSE': cv_rmse}))

# Importância das features
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6,4))
# Ajuste: evitar FutureWarning usando color em vez de palette sem hue
sns.barplot(x=feat_imp.values, y=feat_imp.index, color='tab:blue')
plt.title('Importância das Features'); plt.xlabel('Importância'); plt.tight_layout(); plt.show()
print(feat_imp)

# Bootstrap de resíduos para intervalos preditivos
residuals_rf = y_val - pred_rf
n_boot = 500
boot_preds = []
for _ in range(n_boot):
    sampled = np.random.choice(residuals_rf, size=len(residuals_rf), replace=True)
    boot_preds.append(pred_rf + sampled)
boot_preds = np.array(boot_preds)
pred_interval_lower = np.percentile(boot_preds, 5, axis=0)
pred_interval_upper = np.percentile(boot_preds, 95, axis=0)

plt.figure(figsize=(10,4))
idx = np.arange(len(y_val))
plt.plot(idx, y_val, label='Real', marker='o', ms=3, lw=0.7)
plt.plot(idx, pred_rf, label='Predito', marker='o', ms=3, lw=0.7)
plt.fill_between(idx, pred_interval_lower, pred_interval_upper, color='gray', alpha=0.3, label='Intervalo 90%')
plt.title('Predições vs Intervalos (RF)'); plt.legend(); plt.tight_layout(); plt.show()

# 5. Prescritiva – Função de custo

def cost_function(p, low=9.0, target=10.0, penalty_nonconf=100.0, shape_below=4, coeff_above=1.0):
    if p < low:
        return penalty_nonconf + (low - p) * penalty_nonconf/2
    elif p < target:
        return (target - p)**shape_below * (penalty_nonconf/10)
    else:
        return (p - target)**2 * coeff_above

print('Teste custo:', [ (x, cost_function(x)) for x in [8.5,9,9.5,10,10.5,11]])

# Simulação distribuição para faixa de QTD_CHOC
base = {}
for col in X.columns:
    if col.startswith('VAR_2_'): base[col] = 0
    elif col != 'QTD_CHOC': base[col] = float(df[col].median())

q_min, q_max = df['QTD_CHOC'].min(), df['QTD_CHOC'].max()
q_grid = np.linspace(q_min, q_max, 60)

def simulate_distribution(model, base_row, residuals, qtd_values, n_sim=1000, feature_order=None):
    if feature_order is None:
        raise ValueError('feature_order deve ser fornecido com a lista de colunas do modelo treinado.')
    sims = {}
    for q in qtd_values:
        row = base_row.copy()
        row['QTD_CHOC'] = q
        # garantir todas as colunas presentes
        for col in feature_order:
            if col not in row:
                if col.startswith('VAR_2_'):
                    row[col] = 0
                else:
                    # usa mediana se existir no df, senão 0
                    row[col] = float(df[col].median()) if col in df.columns else 0.0
        X_row = pd.DataFrame([row])[feature_order]
        pred = model.predict(X_row)[0]
        sampled_res = np.random.choice(residuals, size=n_sim, replace=True)
        sims[q] = pred + sampled_res
    return sims

simulations = simulate_distribution(rf, base, residuals_rf, q_grid, n_sim=800, feature_order=list(X.columns))
expected_cost = []; prob_nonconf = []
for q in q_grid:
    weights = simulations[q]
    costs = [cost_function(w) for w in weights]
    expected_cost.append(np.mean(costs))
    prob_nonconf.append(np.mean(weights < 9.0))
expected_cost = np.array(expected_cost); prob_nonconf = np.array(prob_nonconf)
opt_mask = prob_nonconf <= 0.05
idx_map = np.argmin(expected_cost[opt_mask]) if opt_mask.any() else np.argmin(prob_nonconf)
if opt_mask.any():
    idx_map_global = np.where(opt_mask)[0][idx_map]
else:
    idx_map_global = idx_map
q_opt = q_grid[idx_map_global]
print('QTD_CHOC ótimo (restrição <=5% não conformidade):', round(float(q_opt),2))

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(q_grid, expected_cost, color='tab:blue', label='Custo Esperado')
ax1.set_xlabel('QTD_CHOC'); ax1.set_ylabel('Custo Esperado', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.axvline(q_opt, color='green', linestyle='--', label=f'Ótimo (q={q_opt:.2f})')
ax2 = ax1.twinx()
ax2.plot(q_grid, prob_nonconf, color='tab:red', label='Prob <9g')
ax2.set_ylabel('Prob. Não Conformidade', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.axhline(0.05, color='red', linestyle=':', label='Limite 5%')
lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines+lines2, labels+labels2, loc='upper right')
plt.title('Custo vs Risco de Não Conformidade'); plt.tight_layout(); plt.show()

# Cruzamento com 10g
pred_curve = []
for q in q_grid:
    row = base.copy(); row['QTD_CHOC'] = q
    # preencher colunas faltantes
    for col in X.columns:
        if col not in row:
            if col.startswith('VAR_2_'):
                row[col] = 0
            else:
                row[col] = float(df[col].median()) if col in df.columns else 0.0
    X_row = pd.DataFrame([row])[X.columns]
    pred_curve.append(rf.predict(X_row)[0])
pred_curve = np.array(pred_curve)

crossings = []
for i in range(len(pred_curve)-1):
    y1, y2 = pred_curve[i], pred_curve[i+1]
    if (y1-10)*(y2-10) <= 0 and abs(y2-y1) > 1e-12:
        t = (10 - y1)/(y2 - y1)
        q_at_10 = q_grid[i] + t*(q_grid[i+1]-q_grid[i])
        crossings.append(q_at_10)
print('Valores de QTD_CHOC que cruzam 10g (interp.):', [round(float(c),3) for c in crossings])

plt.figure(figsize=(9,4))
plt.plot(q_grid, pred_curve, label='Predição Média', color='navy')
plt.axhline(10, color='black', linestyle='--', label='Peso 10g')
for q in crossings:
    plt.axvline(q, color='red', linestyle=':', alpha=0.7)
    plt.scatter([q],[10], color='red')
plt.xlabel('QTD_CHOC'); plt.ylabel('PESO_BOMBOM (predição)')
plt.title('Predição do Peso vs QTD_CHOC – Cruzamentos 10g')
plt.legend(); plt.tight_layout(); plt.show()

# Exportações padronizadas (substitui quaisquer versões *_corrigido)
# Ajuste: salvar diretamente em outputs/final_delivery para manter pacote de entrega enxuto
outputs_dir = base_dir / 'outputs' / 'final_delivery'
os.makedirs(outputs_dir, exist_ok=True)
metrics_df.to_csv(outputs_dir / 'model_metrics.csv', index=False)
# gerar grade de custo simples (peso vs qtd)
q_min, q_max = df['QTD_CHOC'].min(), df['QTD_CHOC'].max()
q_grid = np.linspace(q_min, q_max, 60)
# Ajuste: construir DataFrame com todas as colunas de X, incluindo dummies
data_dict = {}
for c in X.columns:
    if c == 'QTD_CHOC':
        data_dict[c] = q_grid
    elif c.startswith('VAR_2_'):
        data_dict[c] = np.zeros_like(q_grid)
    elif c in df.columns:
        data_dict[c] = np.full_like(q_grid, fill_value=df[c].median(), dtype=float)
    else:
        data_dict[c] = np.zeros_like(q_grid, dtype=float)
X_lin_grid = pd.DataFrame(data_dict)[X.columns]
pred_curve_lin = lin_reg.predict(X_lin_grid)
# Evitar sobrescrever custo_grade detalhado do pipeline: salvar como custo_grade_linear.csv
pd.DataFrame({'QTD_CHOC': q_grid, 'pred_linear': pred_curve_lin}).to_csv(outputs_dir / 'custo_grade_linear.csv', index=False)
print('Arquivos salvos em outputs/final_delivery: model_metrics.csv, custo_grade_linear.csv')

# Recomendações (impressão)
print('\nRECOMENDAÇÕES EXECUTIVAS:')
print('- Operar QTD_CHOC próximo de', round(float(q_opt),2), 'com monitoramento diário de probabilidade <9g (meta <=5%).')
print('- Reduzir variabilidade de VAR_1: estabilidade diminui dispersão do peso e risco.')
print('- Ajustar política de custo para excesso >10g conforme margens financeiras.')
print('- Recalibrar modelo trimestralmente ou se R2 cair >5 pontos.')
print('- Coletar mais dados em faixa baixa de QTD_CHOC para refinar risco de não conformidade.')
print('- Quase 40% da variação vem da variável de processo (VAR_1). Padronizar VAR_1 e as condições de VAR_2 junto com a dosagem dos outros ingredientes evita desperdício, mantém formato e sabor consistentes e reduz risco de descarte.')
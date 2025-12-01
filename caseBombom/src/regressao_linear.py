import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

np.random.seed(42)

EXPECTED_COLS = ['QTD_CHOC','VAR_1','VAR_2','PESO_BOMBOM']


def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega arquivo CSV com separador ';' e decimal ','. Faz validações básicas."""
    p = Path(caminho)
    if not p.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {p}')
    # Leitura principal
    df = pd.read_csv(p, sep=';', decimal=',')
    df.columns = [c.strip() for c in df.columns]
    faltando = set(EXPECTED_COLS) - set(df.columns)
    if faltando:
        raise ValueError(f'Colunas ausentes: {faltando}. Encontradas: {df.columns.tolist()}')
    # Tipos
    for col in ['QTD_CHOC','VAR_1','PESO_BOMBOM']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['QTD_CHOC','VAR_1','PESO_BOMBOM']].isna().any().any():
        raise ValueError('Falha conversão numérica (verifique separador decimal).')
    df['VAR_2'] = df['VAR_2'].astype('category')
    return df


def preparar_dados(df: pd.DataFrame):
    """One-hot para VAR_2 e split treino/validação."""
    df_model = pd.get_dummies(df, columns=['VAR_2'], drop_first=True)
    X = df_model.drop('PESO_BOMBOM', axis=1)
    y = df_model['PESO_BOMBOM']
    return train_test_split(X, y, test_size=0.25, random_state=42), X.columns.tolist()


def avaliar_modelos(X_train, X_val, y_train, y_val):
    resultados = []
    # Modelo Linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_val)
    mse_lin = mean_squared_error(y_val, pred_lin)
    resultados.append({
        'model': 'Linear',
        'RMSE': mse_lin**0.5,
        'MAE': mean_absolute_error(y_val, pred_lin),
        'R2': r2_score(y_val, pred_lin)
    })
    # Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_val)
    mse_rf = mean_squared_error(y_val, pred_rf)
    resultados.append({
        'model': 'RandomForest',
        'RMSE': mse_rf**0.5,
        'MAE': mean_absolute_error(y_val, pred_rf),
        'R2': r2_score(y_val, pred_rf)
    })
    return resultados, lin, rf, pred_lin, pred_rf


def cross_validation_score(model, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    rmse = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
    return rmse, r2


def resumo_conformidade(df: pd.DataFrame) -> dict:
    """Resumo de conformidade (<9g) e excesso (>10g)."""
    total = len(df)
    abaixo = (df['PESO_BOMBOM'] < 9).sum()
    acima = (df['PESO_BOMBOM'] > 10).sum()
    return {
        'total_registros': total,
        'pct_abaixo_9g': abaixo/total*100,
        'pct_acima_10g': acima/total*100,
        'peso_medio': df['PESO_BOMBOM'].mean()
    }


def salvar_metricas(metrics, path_out: Path):
    path_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(path_out.with_suffix('.csv'), index=False)
    import json
    with open(path_out.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    print('Executando script em:', __file__)
    base_dir = Path(__file__).resolve().parent.parent
    caminho_csv = base_dir / 'data' / 'raw' / 'registros_train.csv'
    df = carregar_dados(str(caminho_csv))
    print(f'Dados carregados: shape={df.shape}')
    print('Colunas:', df.columns.tolist())
    print('Resumo conformidade:', resumo_conformidade(df))
    (X_train, X_val, y_train, y_val), feature_list = preparar_dados(df)
    resultados, lin_model, rf_model, pred_lin, pred_rf = avaliar_modelos(X_train, X_val, y_train, y_val)
    print('\nMétricas Validação:')
    for r in resultados:
        print(r)
    # Cross-validation no melhor modelo (RF)
    df_model = pd.get_dummies(df, columns=['VAR_2'], drop_first=True)
    X_full = df_model.drop('PESO_BOMBOM', axis=1)
    y_full = df_model['PESO_BOMBOM']
    rmse_cv, r2_cv = cross_validation_score(rf_model, X_full, y_full, folds=5)
    print('\nCV RandomForest (RMSE):', rmse_cv.round(4))
    print('CV RandomForest (R2):', r2_cv.round(4))
    # Importância das features RF
    importances = pd.Series(rf_model.feature_importances_, index=feature_list).sort_values(ascending=False)
    print('\nImportâncias (RF):')
    print(importances.round(4))
    # Salvar métricas
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M')
    out_path = base_dir / 'outputs' / run_id / 'model_metrics'
    salvar_metricas(resultados, out_path)
    print(f'Arquivos de métricas salvos em {out_path.parent}')


if __name__ == '__main__':
    main()

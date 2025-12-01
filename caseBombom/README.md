# Case Bombom

## 1. Introdução
Este projeto aplica Advanced Analytics ao processo industrial de produção de bombons para responder quatro perguntas: (1) O que está acontecendo com o peso? (2) Por que ocorre variação? (3) O que vai acontecer? (4) O que fazer para otimizar custo e reduzir risco? O foco é controlar o peso médio `PESO_BOMBOM` por meio da variável controlável `QTD_CHOC`, considerando influência de `VAR_1` e `VAR_2` (não controláveis).

## 2. Problema de Negócio
- Peso alvo: 10 g.
- Abaixo de 9 g: não conformidade (descarte / perda).
- Acima de 10 g: não há limite regulatório, mas há aumento de custo (excesso de matéria-prima).
- Desafio: encontrar setpoint de `QTD_CHOC` que minimiza custo esperado sem elevar o risco de bombons <9 g acima do limite aceitável.

## 3. Dados
Arquivo bruto: `data/raw/registros_train.csv` (500 registros). Colunas:
- `QTD_CHOC` (float) – quantidade de chocolate (controlável)
- `VAR_1` (float) – variável de processo não controlável
- `VAR_2` (categórica) – condição de processo (níveis)
- `PESO_BOMBOM` (float) – variável alvo (peso médio)

Não há valores faltantes; tipos foram normalizados diretamente via leitura CSV (sep=';', decimal=',').

## 4. Metodologia
1. Descritiva: distribuição, estatísticas (CV, IQR), outliers.
2. Diagnóstica: correlações Pearson/Spearman e correlação parcial controlando `QTD_CHOC`.
3. Preditiva: comparação Regressão Linear (baseline) vs Random Forest; avaliação com RMSE, MAE, R² e Cross-Validation.
4. Prescritiva: simulação Monte Carlo (bootstrap de resíduos) para custo esperado e probabilidade de não conformidade; otimização sob restrição de risco (<=5%).

## 5. Função de Custo (Resumo)
```
Se peso < 9: penalidade severa (descarte + adicional)
Se 9 ≤ peso < 10: aproximação ao mínimo com curva acentuada
Se peso ≥ 10: penalização quadrática suave (excesso)
```
Parâmetros ajustáveis em `pipeline_full.py` (`penalty_nonconf`, `shape_below`, `coeff_above`).

## 6. Estrutura de Pastas
```
caseBombom/
  data/
    raw/registros_train.csv
  notebooks/
    Bombom.ipynb        # narrativa analítica completa
    Bombom.py           # versão script exploratório
  src/
    pipeline_full.py    # pipeline principal (único necessário para gerar entrega)
    regressao_linear.py # comparação simples modelos
    test_pipeline_basic.py # teste rápido de integridade dos artefatos
  outputs/
    final_delivery/     # pacote consolidado (artefatos finais)
      metrics/ tables/ figures/ models/ summary.* run_config.json
  README.md
  requirements.txt
```

## 7. Pipeline Principal
Script: `src/pipeline_full.py`
Etapas automatizadas:
- Carregamento e validação de dados
- Estatísticas e correlações
- Treino Linear vs RandomForest + métricas + Cross-Validation
- Simulação custo × risco + ponto ótimo global (risco <=5%)
- Plano por nível de `VAR_1` (produção simulada sob risco)
- Persistência dos artefatos em `outputs/final_delivery/`

### Executar
```bash
python caseBombom/src/pipeline_full.py
```
(Executa, gera/atualiza o pacote final.)

## 8. Artefatos Gerados (final_delivery)
- `metrics/model_metrics.csv` – métricas validação (RMSE, MAE, R²)
- `metrics/cv_metrics.csv` – Cross-Validation RF
- `metrics/prescriptive_optimum.csv` – `QTD_CHOC` ótimo global
- `metrics/production_plan_var1_head.csv` – amostra do plano por VAR_1
- `tables/descriptive.json` – estatísticas, CV, IQR
- `tables/diagnostic_correlations.csv` – correlações ordenadas
- `tables/custo_grade.csv` – custo esperado vs risco vs QTD_CHOC
- `tables/feature_importances.csv` – importâncias (RF)
- `tables/production_plan_var1.csv` – plano completo por nível de VAR_1
- `figures/*.png` – histogramas, importâncias, custo_risco, curva_predicao
- `models/*.pkl` – modelos serializados
- `models/residuals_val.npy` – resíduos validação para simulação futura
- `models/features_median.json` – medianas como template
- `summary.txt / summary.json` – resumo executivo
- `run_config.json` – parâmetros da execução

## 9. Resultados (Exemplo Típico)
- R² Linear ≈ 0.40; RF ≈ 0.38 (varia por execução)
- Importância relativa (RF): `QTD_CHOC` dominante, seguida de `VAR_1`; níveis de `VAR_2` ajustam intercepto.
- Probabilidade de não conformidade (peso <9 g) em ponto ótimo ≤5%
- Ótimo de custo pode não coincidir com predição média = 10g (assimetria de penalização)

## 10. Plano Prescritivo
`production_plan_var1.csv` fornece `QTD_CHOC_opt` para cada nível observado de `VAR_1`, respeitando limite de risco. Uso sugestão:
1. Identificar nível atual/médio de `VAR_1`
2. Selecionar `QTD_CHOC_opt` correspondente
3. Verificar `prob_nonconf`; se próxima ao limite, considerar margem adicional

## 10.1. Recomendações Operacionais 
Principais ações para o gestor reduzir variabilidade e custo:
- Quase 40% da variação explicada vem de uma variável de processo (VAR_1). Padronizar VAR_1 e as condições de VAR_2 junto com a dosagem dos outros ingredientes evita desperdício, mantém formato e sabor consistentes e reduz risco de descarte.
- Monitorar diariamente probabilidade de peso <9 g (meta ≤5%).
- Ajustar `QTD_CHOC` próximo do ótimo calculado, evitando excesso (>10 g) recorrente.
- Revisar parâmetros da função de custo com dados financeiros reais (descarte vs excesso).
- Re-treinar modelo se R² cair mais de 5 pontos ou houver mudança de processo.

## Nota prática (além de QTD_CHOC): Outros fatores também influenciam o peso e a qualidade do bombom.
- Padronizar a dosagem dos demais ingredientes (recheio, cobertura, emulsificantes) e calibrar as balanças.
- Controlar condições de processo (temperatura do molde, tempo de resfriamento) para estabilidade de massa.
- Registrar essas variáveis no dado de produção para incluí-las no modelo e reduzir viés por variáveis omitidas.
- Realizar pequenos testes de experimento (DoE) para quantificar impacto marginal e ajustar o setpoint com mais precisão.

### Indicador Sugerido: Custo por Grama
Calcule e acompanhe mensalmente:
- Custo excedente = (média de gramas acima de 10) × (custo por grama).
- Custo de descarte = (nº de unidades <9 g) × (gramas perdidas) × (custo por grama).
Use estes dois componentes para estimar economia após estabilização.

## 11. Reprodutibilidade
Instalação:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r caseBombom/requirements.txt
python caseBombom/src/pipeline_full.py
```
Teste rápido:
```bash
python caseBombom/src/test_pipeline_basic.py
```

## 12. Próximos Passos
- Incluir modelos Gradient Boosting / XGBoost
- Explorar interações (ex.: `QTD_CHOC * VAR_1`)
- Calibrar função de custo com dados financeiros reais
- Implementar monitoramento de drift (queda significativa em R²)
- Construir API (FastAPI) para recomendação online de setpoint

## 13. Limitações
- Único dataset (500 linhas) limita robustez estatística
- Função de custo simplificada (não incorpora custos marginais reais por grama)
- Tratamento básico de `VAR_2` (apenas one-hot)

---
Para entrega externa simplificada, consulte também `outputs/final_delivery/README_ENTREGA.md`.

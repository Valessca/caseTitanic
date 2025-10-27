# Titanic - Análise e Modelagem

Resumo
- Projeto para predição de sobreviventes do desastre do Titanic usando `Titanic.ipynb`.
- Pré-processamento inclui extração de título a partir da coluna `Name` e imputação de `Age` por títulos e classe.
- Modelos testados: RandomForest, LogisticRegression, KNN, GaussianNB, LinearSVC, SGD, DecisionTree, GradientBoosting (GridSearchCV).

Arquivos principais
- `Titanic.ipynb` — notebook com todo o fluxo: carregamento, tratamento, engenharia de features, treino e geração de submissão.
- `train.csv`, `test.csv` — datasets originais (deve estar na raiz do projeto).
- `titanic_gradient_boosting.csv` — arquivo de saída com predições geradas pelo melhor modelo.

Requisitos
- Python 3.8+
- Principais bibliotecas: `pandas`, `numpy`, `seaborn`, `scikit-learn`
- Recomenda-se criar um ambiente virtual antes de instalar dependências.

Como reproduzir
1. Certifique-se de ter `train.csv` e `test.csv` na raiz do projeto.
2. Execute as células do notebook na ordem.
   - O notebook faz: concatenação dos datasets, extração de `Title` de `Name`, imputação de `Age` por `Title` e `Pclass`, criação de dummies, treino de modelos e validação cruzada.
3. Resultado da predição é salvo em `titanic_gradient_boosting.csv`.

Observações técnicas
- A coluna `Title` é extraída de `Name` com regex e mapeada para categorias comuns (ex: `Mr`, `Mrs`, `Miss`, `Rare`).
- `Age` é preenchida em três passos: mediana por (`Title`, `Pclass`), mediana por `Title`, depois mediana global; convertida para inteiro anulável (`Int64`).
- Função de avaliação: `func_acuracia(algoritmo, x, y, vc)` retorna acurácia de treino e validação cruzada.

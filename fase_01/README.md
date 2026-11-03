# Pipeline preditivo de churn

Projeto do Tech Challenge da Fase 1 (Pós-Tech MLE). Uma operadora de telecomunicações está perdendo clientes e precisa saber quem tem maior risco de cancelar, para agir antes. Este projeto vai da análise exploratória até um modelo servido por uma API REST.

## Resultado

O modelo campeão é uma **regressão logística**, avaliada em 1.409 clientes separados desde o início:

| Métrica no teste | Valor |
|---|---|
| AUC-ROC | **0,841** (IC 95%: 0,818 a 0,862) |
| PR-AUC | 0,628 (a referência sem modelo é 0,265) |
| Recall no limiar 0,26 | 79% de quem cancelou foi identificado |
| Precisão no limiar 0,26 | 50% dos clientes sinalizados cancelaram |

A Random Forest e a MLP não superaram a regressão logística de forma mensurável, então ficamos com o modelo mais simples. Os valores financeiros e as limitações estão no [Model Card](docs/model_card.md).

| Modelo | AUC-ROC (CV) | PR-AUC (CV) | F1 no limiar de negócio | Valor líquido por 1.000 clientes* |
|---|---|---|---|---|
| Referência ingênua (`DummyClassifier`) | 0,500 ± 0,000 | 0,265 | 0,419 | 21.512 |
| **Regressão logística (campeão)** | **0,846 ± 0,014** | **0,661** | **0,630** | **39.278** |
| Random Forest | 0,847 ± 0,010 | 0,662 | 0,605 | 39.150 |
| MLP | 0,840 ± 0,015 | 0,648 | 0,606 | 38.443 |

\* Calculado com premissas ilustrativas, descritas no [ML Canvas](docs/ml_canvas.md). A referência ingênua equivale a ofertar retenção a todos os clientes.

## Como executar

Requer Python 3.12 ou superior (testado no 3.14). Os comandos abaixo partem da pasta `fase_01/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

### Subir a API

O modelo treinado já está em `models/champion.joblib`.

```bash
uvicorn churn.api.main:app --port 8000
```

A documentação interativa fica em <http://localhost:8000/docs>.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65
  }'
```

```json
{"churn_probability": 0.6855, "churn_prediction": true, "threshold": 0.26, "model_name": "logistic_regression"}
```

| Endpoint | Função | Respostas |
|---|---|---|
| `GET /health` | Indica se a API está no ar e se o modelo foi carregado | `200` |
| `POST /predict` | Recebe os dados de um cliente e devolve a propensão de churn | `200`; `422` se os dados forem inválidos; `503` sem modelo |

Para usar outro arquivo de modelo, defina a variável de ambiente `CHURN_MODEL_PATH`.

### Rodar os testes

```bash
pytest
```

São 28 testes, cobrindo limpeza de dados, métricas, regra de seleção, reprodutibilidade e API.

### Treinar novamente

```bash
python -m churn.train
```

O comando leva alguns minutos e sobrescreve o modelo e os resultados em `docs/experiments/`. Duas execuções seguidas geraram resultados idênticos.

## Estrutura

```text
fase_01/
├── data/                  Dataset bruto (Telco Customer Churn)
├── docs/
│   ├── ml_canvas.md       Problema, stakeholders e métricas
│   ├── model_card.md      Desempenho, limitações e vieses
│   ├── roteiro_video_star.md
│   ├── experiments/       Tabela comparativa e métricas do campeão
│   └── figures/           Gráficos gerados pelos notebooks
├── models/champion.joblib Modelo final
├── notebooks/             Experimentação (EDA, baseline e comparação)
├── src/churn/             Código de produção
│   ├── config.py          Caminhos, seed, colunas e premissas
│   ├── data.py            Carga e limpeza, compartilhadas com a API
│   ├── models.py          Pré-processamento e candidatos
│   ├── evaluate.py        Métricas, limiar e seleção do campeão
│   ├── train.py           Fluxo de treino
│   ├── predict.py         Carga do modelo e predição
│   └── api/               FastAPI: rotas e contratos
└── tests/                 Testes com pytest
```

## Decisões técnicas

- **Sem vazamento de dados:** 20% dos dados ficam separados como teste, avaliado uma única vez e só com o campeão. Todo o pré-processamento roda dentro de um `Pipeline`, ajustado apenas em cada fold de treino.
- **Comparação justa:** validação cruzada aninhada (5 folds externos, 3 internos), para que a busca de hiperparâmetros não infle a estimativa de desempenho.
- **Regra de seleção definida no código:** entre os modelos a até um desvio-padrão do melhor em AUC-ROC, vence o mais simples.
- **Propensão utilizável:** nenhum modelo usa `class_weight`, que distorceria as probabilidades devolvidas pela API. O desbalanceamento é tratado no limiar de decisão.
- **Limiar pelo valor de negócio:** escolhido nas predições de validação do treino, não fixado em 0,5.
- **Mesma limpeza no treino e na API:** ambos usam `churn.data.clean_features`.
- **`TotalCharges` em branco vira 0:** os 11 casos são de clientes com `tenure = 0`, que ainda não receberam cobrança.
- **Reprodutibilidade:** seed fixada em `config.py` e dependências com versões exatas em `requirements.txt`.

## Documentação

- [ML Canvas](docs/ml_canvas.md)
- [Model Card](docs/model_card.md)
- [Roteiro do vídeo STAR](docs/roteiro_video_star.md)
- [Análise exploratória](notebooks/EDA.ipynb)
- [Baseline e comparação de modelos](notebooks/02_baseline_e_comparacao.ipynb)

## Dados

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), da IBM, disponível no Kaggle: 7.043 clientes, 19 variáveis explicativas e alvo binário `Churn`.

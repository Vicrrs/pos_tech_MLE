# Model Card: previsão de churn

| | |
|---|---|
| **Modelo** | Regressão logística (`C=10`) em `Pipeline` do scikit-learn 1.9.1 |
| **Versão** | 0.1.0 (2026-11-03) |
| **Arquivo** | `models/champion.joblib` |
| **Saída** | Propensão de churn e decisão no limiar 0,26 |

## Uso pretendido

**Indicado para:** priorizar clientes em campanhas de retenção, com uma pessoa responsável pela estratégia.

**Não indicado para:**

- Negar serviço, piorar condições ou tomar decisões automáticas que prejudiquem o cliente.
- Concluir o que causa o churn. O modelo encontra associações, não causas.
- Aplicar a outras operadoras ou regiões sem nova validação.

## Dados

Telco Customer Churn (IBM/Kaggle): 7.043 clientes e 26,5% de churn. Foram 5.634 clientes para treino e validação e 1.409 para teste, com divisão estratificada e seed 42. `customerID` foi removido.

## Desempenho

### Comparação na validação cruzada (treino)

Validação aninhada com 5 folds externos e 3 internos.

| Modelo | AUC-ROC | PR-AUC | Brier | Limiar | F1 | Precisão | Recall | Valor líquido por 1.000* |
|---|---|---|---|---|---|---|---|---|
| Referência ingênua | 0,500 | 0,265 | 0,195 | 0,05 | 0,419 | 0,265 | 1,000 | 21.512 |
| **Regressão logística** | **0,846 ± 0,014** | **0,661** | **0,135** | **0,26** | **0,630** | **0,517** | **0,808** | **39.278** |
| Random Forest | 0,847 ± 0,010 | 0,662 | 0,135 | 0,19 | 0,605 | 0,456 | 0,897 | 39.150 |
| MLP | 0,840 ± 0,015 | 0,648 | 0,138 | 0,19 | 0,606 | 0,469 | 0,860 | 38.443 |

\* Sob as premissas ilustrativas do [ML Canvas](ml_canvas.md). Limiares, F1, precisão, recall e valor foram medidos nas mesmas predições usadas para escolher o limiar, portanto são levemente otimistas. Os números de teste abaixo não têm esse problema.

**Por que este modelo:** a diferença de 0,001 em AUC para a Random Forest é bem menor que a variação entre folds. Pela regra de seleção, vence o modelo mais simples dentro dessa margem. O ajuste de hiperparâmetros também não trouxe ganho relevante sobre o baseline com valores padrão (AUC de 0,846 nos dois casos).

### Resultado no teste

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,841 (IC 95% por bootstrap: 0,818 a 0,862) |
| PR-AUC | 0,628 |
| Brier | 0,139 |
| F1 | 0,616 |
| Precisão | 0,503 |
| Recall | 0,794 |
| Clientes sinalizados | 41,9% |

| | Previsto: fica | Previsto: cancela |
|---|---:|---:|
| **Ficou** | 741 | 294 |
| **Cancelou** | 77 | 297 |

**Calibração:** a propensão média foi 26,8%, para uma taxa real de 26,5%. As probabilidades acompanham o observado em todas as faixas de risco:

| Faixa de risco (quintil) | Propensão média | Churn observado |
|---|---:|---:|
| 1 (menor) | 1,2% | 1,4% |
| 2 | 6,4% | 7,4% |
| 3 | 19,2% | 21,0% |
| 4 | 41,0% | 37,2% |
| 5 (maior) | 66,1% | 65,6% |

### Valor de negócio (premissas ilustrativas)

No teste, por 1.000 clientes, o modelo gerou **36.605**, contra **19.534** ao ofertar a todos.

Esse ganho depende da taxa de sucesso da retenção, que é uma suposição. Mantendo o limiar em 0,26:

| Taxa de sucesso | Valor líquido por 1.000 clientes |
|---:|---:|
| 10% | **−1.780** (prejuízo) |
| 20% | 17.413 |
| 30% | 36.605 |
| 50% | 74.990 |

Abaixo de cerca de 11% de sucesso, a campanha nesse limiar dá prejuízo.

## Limitações

- **Premissas financeiras não validadas.** Custo da oferta, taxa de sucesso e meses retidos são ilustrativos, e o limiar 0,26 depende deles.
- **Risco não é o mesmo que capacidade de retenção.** O modelo não diz quem mudará de ideia com uma oferta. Isso exigiria um experimento controlado e um modelo de uplift.
- **Sem validação temporal.** O dataset é uma fotografia sem datas, então não foi possível testar o desempenho em períodos futuros. O resultado real pode ser pior.
- **Possível vazamento temporal.** `tenure` e `TotalCharges` podem ter sido registrados no momento do cancelamento. Em produção, devem refletir o estado do cliente **antes** do período previsto.
- **Dataset de demonstração.** É uma base pública de exemplo da IBM, com uma única operadora e sem indicação de país ou moeda. O desempenho em dados reais precisa ser medido.
- **Variáveis ausentes.** Não há dados de uso, reclamações, qualidade da rede ou concorrência, que provavelmente também influenciam o churn.
- **Ordenação depende muito do contrato.** Dentro de cada tipo de contrato, a AUC cai para cerca de 0,74 a 0,76. Parte do bom resultado geral vem de separar os tipos de contrato, que a operadora já conhece.
- **Coeficientes não são efeitos causais.** `MonthlyCharges` tem coeficiente negativo, embora quem paga mais cancele mais, por causa da colinearidade com os serviços contratados. Isso não indica que aumentar o preço reduz o churn.
- **Sem monitoramento.** O projeto ainda não mede mudança de distribuição nem queda de desempenho em produção.

## Possíveis vieses

Métricas no conjunto de teste, no limiar 0,26. Tabela completa em [`subgroup_metrics.csv`](experiments/subgroup_metrics.csv).

| Grupo | n | Churn real | AUC-ROC | Recall | Precisão | Sinalizados |
|---|---:|---:|---:|---:|---:|---:|
| Feminino | 687 | 28,1% | 0,836 | 0,788 | 0,524 | 42,2% |
| Masculino | 722 | 25,1% | 0,847 | 0,801 | 0,482 | 41,7% |
| Não idoso | 1.187 | 23,2% | 0,842 | 0,746 | 0,470 | 36,9% |
| **Idoso** | **222** | **44,1%** | **0,782** | **0,929** | **0,595** | **68,9%** |
| Sem dependentes | 978 | 31,0% | 0,825 | 0,828 | 0,505 | 50,8% |
| Com dependentes | 431 | 16,5% | 0,855 | 0,648 | 0,489 | 21,8% |
| Mês a mês | 773 | 42,6% | 0,742 | 0,878 | 0,518 | 72,2% |
| Um ano | 300 | 12,0% | 0,738 | 0,222 | 0,242 | 11,0% |
| Dois anos | 336 | 2,7% | 0,761 | 0,000 | 0,000 | 0,0% |

**O que observamos:**

- **Gênero:** desempenho e taxa de sinalização semelhantes.
- **Idosos:** o modelo ordena pior (AUC de 0,78 contra 0,84) e sinaliza 69% do grupo, contra 37% dos demais. A taxa real de churn também é maior, mas o grupo é pequeno no teste (222 pessoas), o que torna as estimativas menos estáveis.
- **Com dependentes:** recall menor (65% contra 83%), então mais cancelamentos passam sem ser detectados.
- **Contratos longos:** o modelo praticamente não encontra quem cancela. Nos contratos de dois anos, não sinalizou ninguém, deixando passar os 9 cancelamentos do teste.

**Efeito prático:** se a oferta for um benefício, ser sinalizado com mais frequência favorece alguns grupos e deixa outros de fora. Se a abordagem for vista como incômoda, o efeito se inverte. A análise cobre apenas os atributos disponíveis, e não há raça, renda ou região no dataset.

**Atributos sensíveis:** retirar `gender` e `SeniorCitizen` praticamente não muda a AUC na validação (0,8464 com, 0,8465 sem). Por isso, **recomendamos remover esses atributos na próxima versão**, salvo razão de negócio ou legal para mantê-los. Isso sozinho não elimina as diferenças entre grupos, porque outras variáveis podem funcionar como substitutas, então as métricas por subgrupo devem continuar sendo acompanhadas. O uso de dados pessoais também deve ser avaliado à luz da LGPD.

## Recomendações antes de usar em produção

1. Substituir as premissas financeiras pelos valores reais e recalcular o limiar.
2. Rodar um teste A/B com grupo de controle, para medir o efeito real das ofertas.
3. Validar com dados recentes da operadora, respeitando a ordem temporal.
4. Implantar monitoramento de distribuição, desempenho e métricas por subgrupo.
5. Decidir sobre a remoção de `gender` e `SeniorCitizen`.

# ML Canvas: previsão de churn

## Problema de negócio

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado: 26,5% da base do dataset cancelou. A retenção hoje não tem como priorizar quem abordar. O projeto entrega uma lista ordenada por risco, para concentrar o orçamento nos clientes com maior chance de sair.

## Stakeholders

| Stakeholder | Interesse | Como usa o resultado |
|---|---|---|
| Diretoria | Reduzir a perda de receita | Acompanha a métrica de negócio e aprova o orçamento |
| Time de retenção (CRM) | Saber quem abordar | Consome a propensão para montar campanhas |
| Marketing e produto | Entender os fatores de risco | Investiga contrato, fibra e suporte técnico |
| Engenharia e ML | Operar e evoluir o modelo | Mantém a API e monitora o desempenho |
| Clientes | Não ser tratados de forma injusta | São afetados pela oferta ou pela falta dela |

## Tarefa de ML

| Item | Definição |
|---|---|
| Tipo | Classificação binária supervisionada |
| Alvo | `Churn`: cliente saiu no último mês (Yes/No) |
| Entrada | Dados do cliente: perfil, serviços, contrato e cobrança |
| Saída | Propensão de churn (0 a 1) e decisão de abordar ou não |
| Uso | Sob demanda via API; também pode pontuar a base inteira em lote |

## Dados

- **Fonte:** Telco Customer Churn (IBM/Kaggle), com 7.043 clientes e 19 variáveis explicativas.
- **Qualidade:** sem duplicatas. Há apenas 11 `TotalCharges` em branco, todos de clientes sem cobrança ainda.
- **Divisão:** 80% para treino e validação cruzada, 20% para teste, com estratificação e seed fixa.

## Métricas

### Técnicas

- **Principal: AUC-ROC.** Mede a qualidade da ordenação por risco, que é o que uma campanha com orçamento limitado precisa.
- **Complementares:** PR-AUC (sensível à classe minoritária), F1, precisão, recall e Brier (qualidade das probabilidades).
- **Não usada: acurácia.** Prever que ninguém cancela já acertaria 73,5%.

### Negócio

**Valor líquido da campanha por 1.000 clientes:** receita preservada menos o custo das ofertas.

```text
valor líquido = soma, nos clientes sinalizados, de:
    (churn real) × taxa de sucesso × meses retidos × mensalidade − custo da oferta
```

| Premissa | Valor | Origem |
|---|---|---|
| Custo por oferta | 50 | **Ilustrativa** |
| Taxa de sucesso da retenção | 30% | **Ilustrativa** |
| Meses de receita preservados | 12 | **Ilustrativa** |

> **Atenção:** o dataset não traz nenhum desses valores. Eles servem para demonstrar o método e precisam ser substituídos pelos números reais da operadora em `src/churn/config.py`. O resultado é sensível a eles; veja o [Model Card](model_card.md). O valor também não inclui clientes que seriam retidos sem a oferta, nem efeitos da oferta sobre quem não pretendia sair.

## Critérios de sucesso

| Critério | Meta | Resultado no teste |
|---|---|---|
| AUC-ROC acima da referência ingênua (0,50) | Superar com folga | 0,841 |
| Valor líquido acima de ofertar a todos | Superar | 36.605 contra 19.534 por 1.000 clientes |
| API funcional com `/health` e `/predict` | Funcionar | Testada com pytest e com servidor em execução |

## Ações e riscos

| Risco | Mitigação |
|---|---|
| Premissas financeiras incorretas | Substituir pelos valores reais e validar com teste A/B |
| Mudança no perfil da base | Monitorar desempenho e distribuição das variáveis; retreinar quando necessário |
| Tratamento desigual entre grupos | Acompanhar métricas por subgrupo; revisar o uso de atributos sensíveis |
| Confundir correlação com causa | Tratar os fatores de risco como hipóteses a testar |

# Roteiro do vídeo STAR (até 5 minutos)

Fala de cerca de 4min30, deixando folga. Antes de gravar, deixe abertos o README, os gráficos e um terminal na pasta `fase_01/` com o ambiente ativado.

## Situation: 45 segundos

**Na tela:** README e `docs/figures/churn_by_category.png`.

> Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado, e a diretoria precisa saber quem tem maior risco de cancelar, para agir antes que isso aconteça.
>
> Usamos o dataset Telco Customer Churn, da IBM, com pouco mais de 7 mil clientes. Ele traz perfil, serviços contratados, tipo de contrato e cobrança. Cerca de 26,5% dos clientes cancelaram.
>
> Na análise exploratória, o contrato mês a mês se destacou, com 43% de churn contra 3% no contrato de dois anos. O risco também é maior nos primeiros meses: 53% de churn até o sexto mês.

## Task: 30 segundos

**Na tela:** `docs/ml_canvas.md`.

> Nossa tarefa foi construir o caminho completo: da análise até um modelo servido por API, comparando um modelo linear, um de árvores e uma rede neural.
>
> Definimos duas métricas. A técnica é a AUC-ROC, porque a campanha precisa de uma boa ordenação por risco. Não usamos acurácia, já que prever que ninguém cancela acertaria 73%. A de negócio é o valor líquido da campanha de retenção. Os custos e a taxa de sucesso são premissas ilustrativas, pois não estão no dataset.

## Action: 2 minutos

**Na tela:** `notebooks/EDA.ipynb`, na célula de qualidade dos dados.

> Na limpeza, encontramos 11 valores em branco em TotalCharges. Todos eram de clientes com zero meses de casa, que ainda não tinham recebido cobrança. Então preenchemos com zero, e não com a média, porque esse é o valor real.

**Na tela:** `docs/figures/model_comparison.png`.

> Para comparar os modelos, separamos 20% dos dados como teste, usado uma única vez no final. No restante, aplicamos validação cruzada aninhada, para que a busca de hiperparâmetros não inflasse os resultados. Todo o pré-processamento fica dentro de um Pipeline, o que evita vazamento de dados.
>
> Os três modelos empataram: Random Forest com AUC de 0,847, regressão logística com 0,846 e MLP com 0,840. A diferença é menor que a variação entre os folds. Nossa regra foi escolher o modelo mais simples dentro dessa margem, e a regressão logística venceu. Ela é mais barata e mais fácil de explicar.

**Na tela:** `docs/figures/threshold_value.png`.

> Também não usamos o limiar padrão de 0,5. Escolhemos o limiar que maximiza o valor de negócio, que ficou em 0,26. Como não usamos pesos de classe, a API devolve probabilidades que acompanham o churn observado.

**Na tela:** árvore de pastas no editor e `src/churn/api/main.py`.

> Depois levamos o código dos notebooks para módulos em `src`, com responsabilidades separadas. A mesma função de limpeza é usada no treino e na API, evitando diferenças entre os dois. A API, em FastAPI, tem os endpoints `health` e `predict`, e valida a entrada: rejeita valores inválidos e combinações impossíveis, como um cliente sem internet com suporte técnico contratado.

## Result: 1min15

**Na tela:** terminal.

```bash
pytest
uvicorn churn.api.main:app --port 8000
```

> Os 28 testes passam. Agora vou subir a API.

**Na tela:** <http://localhost:8000/docs>. Execute `/health` e depois `/predict` com o exemplo pronto.

> O `health` confirma que a API está no ar com o modelo carregado. No `predict`, envio um cliente com dois meses de casa, contrato mês a mês e fibra. A propensão de churn fica alta e ele é sinalizado para retenção.

**Na tela:** seção de resultados e vieses em `docs/model_card.md`.

> No teste, a AUC foi de 0,841, próxima da validação, o que indica boa generalização. Identificamos 79% de quem cancelou, com precisão de 50%. Com nossas premissas, o valor por mil clientes foi de cerca de 36 mil, contra 19 mil ao ofertar a todos.
>
> O Model Card registra as limitações. O valor financeiro depende das premissas: com 10% de sucesso, a campanha dá prejuízo. O dataset não tem datas, então não há validação temporal. E o modelo sinaliza 69% dos idosos, contra 37% dos demais. Retirar gênero e a indicação de idoso praticamente não altera a AUC, então recomendamos remover esses atributos na próxima versão.
>
> Como próximo passo, a operadora deve validar as premissas com um teste A/B. Assim, o projeto entrega uma API funcional, testada e reprodutível, com clareza sobre onde o modelo pode ou não ser usado.

## Antes de enviar

- [ ] Cronometrar: o vídeo precisa ter no máximo 5 minutos.
- [ ] Ajustar "nós" para "eu" se a entrega for individual.
- [ ] Mostrar a API respondendo de verdade, não só slides.
- [ ] Conferir se os números falados batem com o README.

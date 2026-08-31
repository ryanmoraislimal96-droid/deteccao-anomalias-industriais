# 🏭 Detecção de Anomalias Industriais com Machine Learning

Sistema inteligente para análise de dados de sensores industriais, identificação de padrões associados a falhas e apoio à manutenção preventiva.

O projeto compara uma abordagem não supervisionada com Isolation Forest e uma abordagem supervisionada com Random Forest, além de incorporar explicabilidade com SHAP, validação de leituras e uma interface interativa construída com Gradio.

---

## 🎯 Objetivo

Desenvolver um sistema capaz de analisar leituras de sensores de equipamentos industriais e identificar comportamentos associados a possíveis falhas.

O sistema busca não apenas realizar previsões, mas também explicar os principais fatores que influenciaram cada decisão.

---

## 📊 Dataset

Foi utilizado o **AI4I 2020 Predictive Maintenance Dataset**, composto por 10.000 registros de funcionamento de equipamentos industriais.

A variável alvo utilizada foi:

- `Machine failure`

Foram utilizados cinco sensores como entrada do modelo:

- 🌡️ Temperatura do ar
- 🔥 Temperatura do processo
- ⚙️ Velocidade de rotação
- 🔩 Torque
- 🛠️ Desgaste da ferramenta

O conjunto apresenta forte desbalanceamento:

- Operações normais: **9.661 (96,61%)**
- Falhas: **339 (3,39%)**

Por esse motivo, métricas como Precision, Recall e F1-score foram priorizadas em relação à acurácia.

---

## 🤖 Modelos avaliados

### Isolation Forest

Utilizado como baseline de detecção de anomalias não supervisionada.

O modelo apresentou desempenho limitado para identificar especificamente as falhas reais do dataset.

### Random Forest V5

A solução final utiliza uma Random Forest supervisionada com:

- `200` árvores
- `class_weight="balanced"`
- `random_state=42`
- limite de decisão ajustado para **30%**

O limite foi escolhido utilizando exclusivamente o conjunto de validação.

---

## 🏆 Resultados finais

A avaliação final foi realizada em um conjunto de teste separado, contendo 2.000 registros.

| Métrica | Resultado |
|---|---:|
| Precision | **64,86%** |
| Recall | **70,59%** |
| F1-score | **67,61%** |

### Matriz de confusão

- ✅ 1.906 operações normais classificadas corretamente
- 🚨 26 falsos alertas
- ⚠️ 20 falhas não detectadas
- ✅ 48 falhas detectadas corretamente

---

## 🎚️ Ajuste do limite de decisão

O limite padrão de 50% apresentou:

- Precision: **80,00%**
- Recall: **41,18%**
- F1-score: **54,37%**

Ao reduzir o limite para 30%, o sistema passou a detectar uma quantidade maior de falhas reais.

O objetivo foi encontrar um equilíbrio entre sensibilidade e geração de falsos alertas.

---

## 🧠 Inteligência Artificial Explicável

O projeto utiliza **SHAP** para explicar previsões individuais.

Em um exemplo com risco previsto de **87%**, os principais fatores que elevaram o score de falha foram:

- 🔩 Torque
- ⚙️ Rotação
- 🔥 Temperatura do processo

Também foi analisada a importância global das variáveis:

| Sensor | Importância |
|---|---:|
| Torque | 34,29% |
| Rotação | 29,28% |
| Desgaste da ferramenta | 19,66% |
| Temperatura do ar | 10,45% |
| Temperatura do processo | 6,32% |

---

## 🛡️ Validação das leituras

Antes de realizar uma previsão, o sistema verifica se as leituras estão dentro das faixas conhecidas durante o treinamento.

Caso um valor esteja fora do domínio observado, a análise é bloqueada.

Exemplo:

```text
Rotação: 50000 RPM

ANÁLISE NÃO REALIZADA

A leitura está fora da faixa conhecida pelo modelo.

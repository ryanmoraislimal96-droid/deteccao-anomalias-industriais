import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import gradio as gr


# ============================================
# CARREGAR MODELO
# ============================================

pacote = joblib.load(
    "modelo_anomalias_industriais_v5.pkl"
)

modelo = pacote["modelo"]
limite = pacote["limite"]
sensores = pacote["sensores"]
faixas = pacote["faixas"]


nomes_pt = {
    "Air temperature": "Temperatura do ar",
    "Process temperature": "Temperatura do processo",
    "Rotational speed": "Rotação",
    "Torque": "Torque",
    "Tool wear": "Desgaste da ferramenta"
}


# SHAP específico para modelos baseados em árvores
explainer = shap.TreeExplainer(modelo)


# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def analisar(
    temperatura_ar,
    temperatura_processo,
    rotacao,
    torque,
    desgaste
):

    valores = [
        temperatura_ar,
        temperatura_processo,
        rotacao,
        torque,
        desgaste
    ]

    # ----------------------------------------
    # VALIDAÇÃO DAS ENTRADAS
    # ----------------------------------------

    problemas = []

    for sensor, valor in zip(sensores, valores):

        minimo = faixas[sensor]["minimo"]
        maximo = faixas[sensor]["maximo"]

        if valor < minimo or valor > maximo:

            nome = nomes_pt.get(sensor, sensor)

            problemas.append(
                f"⚠️ **{nome}: {valor}** está fora da "
                f"faixa conhecida ({minimo:.1f} a {maximo:.1f})"
            )

    tabela = pd.DataFrame({
        "Sensor": [
            nomes_pt.get(sensor, sensor)
            for sensor in sensores
        ],
        "Valor": valores
    })

    if problemas:

        resultado = """
# 🛑 ANÁLISE NÃO REALIZADA

Um ou mais sensores estão fora da faixa conhecida pelo modelo.

""" + "\n\n".join(problemas) + """

### 🔧 Recomendação

Verifique as leituras dos sensores antes de realizar uma nova análise.
"""

        return resultado, tabela, "", None


    # ----------------------------------------
    # PREVISÃO
    # ----------------------------------------

    entrada = pd.DataFrame(
        [valores],
        columns=sensores
    )

    score = modelo.predict_proba(
        entrada
    )[0, 1]


    # ----------------------------------------
    # NÍVEL DE RISCO
    # ----------------------------------------

    if score < limite:

        status = "✅ OPERAÇÃO NORMAL"
        nivel = "🟢 BAIXO"

        recomendacao = (
            "Nenhuma anomalia relevante foi identificada. "
            "Mantenha o monitoramento normal do equipamento."
        )

    elif score < 0.50:

        status = "⚠️ ANOMALIA DETECTADA"
        nivel = "🟡 ATENÇÃO"

        recomendacao = (
            "Recomendada verificação preventiva das leituras "
            "e acompanhamento do equipamento."
        )

    elif score < 0.70:

        status = "⚠️ ANOMALIA DETECTADA"
        nivel = "🟠 ALTO"

        recomendacao = (
            "Recomendada avaliação da equipe de manutenção."
        )

    else:

        status = "🚨 ANOMALIA DETECTADA"
        nivel = "🔴 CRÍTICO"

        recomendacao = (
            "Recomendada avaliação prioritária "
            "pela equipe de manutenção."
        )


    resultado = f"""
# {status}

### Nível de risco: {nivel}

## Score do modelo: {score * 100:.2f}%

### 🔧 Recomendação

{recomendacao}
"""


    # ========================================
    # SHAP
    # ========================================

    valores_shap = explainer.shap_values(
        entrada
    )

    if isinstance(valores_shap, list):

        contribuicoes = valores_shap[1][0]

    elif np.asarray(valores_shap).ndim == 3:

        contribuicoes = np.asarray(
            valores_shap
        )[0, :, 1]

    else:

        contribuicoes = np.asarray(
            valores_shap
        )[0]


    df_shap = pd.DataFrame({
        "Sensor": [
            nomes_pt.get(sensor, sensor)
            for sensor in sensores
        ],
        "Impacto SHAP": contribuicoes * 100
    })

    df_shap["Impacto absoluto"] = (
        df_shap["Impacto SHAP"].abs()
    )

    df_shap = df_shap.sort_values(
        "Impacto absoluto",
        ascending=False
    )


    # ----------------------------------------
    # EXPLICAÇÃO EM TEXTO
    # ----------------------------------------

    explicacao = """
## 🧠 Principais fatores desta decisão

"""

    for _, linha in df_shap.head(3).iterrows():

        impacto = linha["Impacto SHAP"]

        if impacto >= 0:
            efeito = "⬆️ aumentou o score de falha"
        else:
            efeito = "⬇️ reduziu o score de falha"

        explicacao += (
            f"**{linha['Sensor']}**: "
            f"{impacto:+.2f} pontos • "
            f"{efeito}\n\n"
        )


    # ----------------------------------------
    # GRÁFICO
    # ----------------------------------------

    grafico = df_shap.sort_values(
        "Impacto SHAP",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        grafico["Sensor"],
        grafico["Impacto SHAP"]
    )

    ax.axvline(0)

    ax.set_xlabel(
        "Impacto SHAP no score de falha"
    )

    ax.set_title(
        f"Explicação da previsão - Risco {score * 100:.0f}%"
    )

    plt.tight_layout()


    return (
        resultado,
        tabela,
        explicacao,
        fig
    )


# ============================================
# INTERFACE
# ============================================

with gr.Blocks(
    title="Detecção de Anomalias Industriais"
) as app:

    gr.Markdown(
        """
# 🏭 Detecção de Anomalias Industriais

### 🌲 Random Forest V5 + 🧠 SHAP + 🛡️ Validação de sensores

Insira as leituras dos sensores para avaliar o equipamento.

**Limite de detecção: 30%**
"""
    )

    temperatura_ar = gr.Number(
        label="🌡️ Temperatura do ar (K)",
        value=298.9
    )

    temperatura_processo = gr.Number(
        label="🔥 Temperatura do processo (K)",
        value=310.2
    )

    rotacao = gr.Number(
        label="⚙️ Rotação (RPM)",
        value=2737
    )

    torque = gr.Number(
        label="🔩 Torque (Nm)",
        value=8.8
    )

    desgaste = gr.Number(
        label="🛠️ Desgaste da ferramenta (min)",
        value=142
    )

    botao = gr.Button(
        "🔍 Analisar equipamento",
        variant="primary"
    )

    resultado = gr.Markdown()

    tabela = gr.Dataframe(
        label="📊 Leituras analisadas",
        interactive=False
    )

    explicacao = gr.Markdown()

    grafico = gr.Plot(
        label="🧠 Explicação da decisão"
    )

    botao.click(
        fn=analisar,
        inputs=[
            temperatura_ar,
            temperatura_processo,
            rotacao,
            torque,
            desgaste
        ],
        outputs=[
            resultado,
            tabela,
            explicacao,
            grafico
        ]
    )


if __name__ == "__main__":
    app.launch()

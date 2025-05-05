import streamlit as st
import pandas as pd
import joblib
import unicodedata
import os
from io import BytesIO


@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("classificador_treinado.pkl")

def limpa_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if c.isalnum() or c.isspace())
    return " ".join(texto.split())

st.title("Classificador TI vs NÃO TI")

uploaded_file = st.file_uploader("Envie sua planilha", type=["xls","xlsx","csv"])
col = st.text_input("Nome da coluna de descrição", value="descricao")

if uploaded_file is not None and col:

    # lê planilha
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    if col not in df.columns:
        st.error(f"Coluna '{col}' não encontrada na planilha. Verifique o nome e tente novamente.")
    else:
        # pré-processa e classifica
        df["_texto_limpo"] = df[col].apply(limpa_texto)
        model = load_model()

        df["_previsto"] = model.predict(df["_texto_limpo"])
        df["_previsto"] = df["_previsto"].map({1: "TI", 0: "NÃO TI"})

        # monta nome de saída a partir do nome de entrada
        base = os.path.splitext(uploaded_file.name)[0]

        # botão de download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar classificado",
            data=csv,
            file_name=f"{base}_classificado.csv",
            mime="text/csv"
        )

        # prepara buffer Excel em memória
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Classificado")
        buffer.seek(0)

        # botão de download Excel
        st.download_button(
            label="📥 Baixar classificado (Excel)",
            data=buffer,
            file_name=f"{base}_classificado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


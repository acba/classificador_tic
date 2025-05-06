import streamlit as st
import pandas as pd
import joblib
import unicodedata
import os
from io import BytesIO
import glob
import zipfile

@st.cache_resource
def load_model(path: str):
    """Carrega e armazena em cache o pipeline pré-treinado."""
    return joblib.load(path)

def limpa_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if c.isalnum() or c.isspace())
    return " ".join(texto.split())

st.title("Classificador TI vs NÃO TI")

model_dir = "classificadores"
model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
if not model_files:
    st.error("Nenhum modelo (.pkl) encontrado e disponível'.")
    st.stop()

# Caixa de seleção para o usuário escolher qual modelo carregar
# Exibe somente o nome do arquivo, sem o caminho completo
model_names = [os.path.basename(f) for f in model_files]
selected_name = st.selectbox("Selecione o modelo a ser usado", model_names)
selected_model = os.path.join(model_dir, selected_name)

base_name = os.path.splitext(selected_model)[0]

# 3) Exibe o conteúdo do arquivo de log associado, se existir
log_path = base_name + ".log"
if os.path.isfile(log_path):
    with st.expander(f"Mostrar log de treinamento ({os.path.basename(log_path)})", expanded=False):
        with open(log_path, "r", encoding="utf-8") as lf:
            log_content = lf.read()
        st.text_area("Log de treinamento:", log_content, height=300)
else:
    st.warning(f"Nenhum arquivo de log encontrado para este modelo ({os.path.basename(log_path)}).")

# 4) Exibe as premières linhas do arquivo _bd.xlsx associado, se existir
bd_path = base_name + "_bd.xlsx"
if os.path.isfile(bd_path):
    with st.expander(f"Prévia dos dados usados no treinamento do classificador ({os.path.basename(bd_path)})", expanded=False):
        df_bd = pd.read_excel(bd_path)
        st.dataframe(df_bd.head())
else:
    st.warning(f"Nenhuma base de dados _bd encontrada ({os.path.basename(bd_path)}).")

# 5) Botão para download conjunto (modelo, log e _bd)
if st.button("📦 Baixar recursos do modelo"):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        # adiciona modelo
        z.write(selected_model, arcname=os.path.basename(selected_model))
        # adiciona log
        if os.path.isfile(log_path):
            z.write(log_path, arcname=os.path.basename(log_path))
        # adiciona base _bd
        if os.path.isfile(bd_path):
            z.write(bd_path, arcname=os.path.basename(bd_path))
    buffer.seek(0)
    st.download_button(
        label="⬇️ Baixar ZIP com modelo, log e _bd",
        data=buffer,
        file_name=f"{os.path.splitext(selected_name)[0]}_resources.zip",
        mime="application/zip"
    )

uploaded_file = st.file_uploader("Envie sua planilha", type=["xls","xlsx","csv"])
col = st.text_input("Nome da coluna de descrição", value="descricao")

if uploaded_file is not None and col:

    if st.button("Classificar 😊"):
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
            model = load_model(selected_model)

            df["_previsto"] = model.predict(df["_texto_limpo"])
            df["_previsto"] = df["_previsto"].map({1: "TI", 0: "NÃO TI"})

            # monta nome de saída a partir do nome de entrada
            base = os.path.splitext(uploaded_file.name)[0]

            # botão de download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Baixar classificado (CSV)",
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


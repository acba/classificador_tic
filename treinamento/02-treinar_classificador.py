#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import shutil
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Cria diretório para salvar modelos, se não existir
modelos_path = 'modelos/'
os.makedirs(modelos_path, exist_ok=True)
nome_modelo = f"{timestamp}_classificador_treinado.pkl"

# redireciona stdout para console + log
f = open(f"{modelos_path}{timestamp}_classificador_treinado.log", "a", encoding="utf-8")
sys.stdout = Tee(sys.stdout, f)


def limpa_texto(texto: str) -> str:
    """
    Normaliza, remove acentos, pontuação e excesso de espaços.
    """
    import unicodedata
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
    texto = ' '.join(texto.split())
    return texto


# 1) Carregar o DataFrame
df_filepath = 'base_dados_balanceado.xlsx'
shutil.copy(df_filepath, f"{modelos_path}{timestamp}_classificador_treinado_bd.xlsx")

df = pd.read_excel(df_filepath)
df.rename(columns={'TIC': 'classe'}, inplace=True)
df['classe'].replace({'Sim': 1, 'Não': 0}, inplace=True)

print("Base de dados :", df_filepath)
print("Tamanho da base de dados :", len(df))
print("Distribuição :", df['classe'].value_counts())
print()

# 2) Pré-processamento
df = df.dropna(subset=['descricao', 'classe']).copy()
df['texto_limpo'] = df['descricao'].apply(limpa_texto)

# 3) Divisão treino/teste
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    stratify=df['classe'],
    random_state=42
)

X_train = df_train['texto_limpo']
y_train = df_train['classe']
X_test  = df_test['texto_limpo']
y_test  = df_test['classe']

# 4) Pipeline TF–IDF + Logistic Regression
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8)
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'))
])

# 5) Busca em grade de hiperparâmetros (grid search)
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 20, 30, 50, 60, 65, 70, 75, 80, 85, 90, 100, 1000],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',     # foco em F1 para equilibrar precisão e recall
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
print("Melhor combinação de hiperparâmetros:", grid.best_params_)
print(f"Melhor F1 (CV): {grid.best_score_:.4f}")

# 6) Avaliação no conjunto de teste com threshold default (0.5)
modelo = grid.best_estimator_
y_pred = modelo.predict(X_test)

print("\n=== Resultados com conjunto de teste ===\n")
print("\nRelatório de classificação (threshold=0.5) ===")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Matriz de confusão === \n", cm)
print()
print(f"True Negatives (TN) número de instâncias reais “NÃO TI” que o modelo também previu como “NÃO TI”: {tn}")
print(f"Falso Positivo (FP) número de instâncias reais “NÃO TI” que o modelo previu como “TI” (falsos alarmes):  {fp}")
print(f"Falso Negativo (FN) número de instâncias reais “TI” que o modelo previu como “NÃO TI” (perdas): {fn} ")
print(f"True Positives (TP) número de instâncias reais “TI” que o modelo corretamente previu como “TI”: {tp} \n")

print("                Previsto NÃO TI   Previsto TI")
print(f"Real NÃO TI    {tn:>15d}   {fp:>11d}")
print(f"Real TI        {fn:>15d}   {tp:>11d}")


# 7) Exibir os 20 primeiros exemplos do teste
resultados = df_test[['descricao', 'classe']].copy()
resultados['previsto_0_5'] = y_pred
# converte para label original, se desejar
resultados['esperado_label'] = resultados['classe'].map({1: 'TI', 0: 'NÃO TI'})
resultados['previsto_label'] = resultados['previsto_0_5'].map({1: 'TI', 0: 'NÃO TI'})

# print("\n--- 20 primeiros exemplos do teste ---")
# print(
#     resultados[['descricao', 'esperado_label', 'previsto_label']]
#     .head(20)
#     .to_string(index=False)
# )

# 9) Exibir 20 primeiros exemplos INCORRETOS
incorretos = resultados[resultados['esperado_label'] != resultados['previsto_label']]
print(f"\n--- {min(len(incorretos), 20)} primeiros exemplos onde previsto != esperado ---")
print(
    incorretos[['descricao', 'esperado_label', 'previsto_label']]
    .head(20)
    .to_markdown(index=False, headers='keys', tablefmt='psql')
)

# 5) Salva o pipeline completo
joblib.dump(modelo, f'{modelos_path}{nome_modelo}')
print(f"\nModelo salvo em: {modelos_path}{nome_modelo}")
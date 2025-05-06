#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

def limpa_texto(texto: str) -> str:
    """
    Normaliza, remove acentos, pontuação e excesso de espaços.
    """
    import unicodedata
    # lower case
    texto = texto.lower()
    # separa acentos
    texto = unicodedata.normalize('NFKD', texto)
    # remove caracteres não alfanuméricos (exceto espaço)
    texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
    # collapse múltiplos espaços
    texto = ' '.join(texto.split())
    return texto

def main():
    parser = argparse.ArgumentParser(
        description="Classifica descrições de uma planilha usando um modelo pré-treinado"
    )
    parser.add_argument(
        "--planilha",
        help="Caminho para o arquivo da planilha (xls, xlsx ou csv)"
    )
    parser.add_argument(
        "--coluna",
        help="Nome da coluna que contém o texto a ser classificado"
    )
    parser.add_argument(
        "--modelo",
        default="classificador_treinado.pkl",
        help="Caminho para o arquivo do modelo serializado (padrão: classificador_treinado.pkl)"
    )
    parser.add_argument(
        "--saida",
        help="Caminho de saída para salvar a planilha classificada (por padrão adiciona '_classificado' antes da extensão)"
    )
    args = parser.parse_args()

    # Verifica existência do modelo
    if not os.path.isfile(args.modelo):
        sys.exit(f"Erro: modelo não encontrado em '{args.modelo}'")

    # Carrega o modelo (pipeline completo)
    modelo = joblib.load(args.modelo)

    # Carrega a planilha
    ext = os.path.splitext(args.planilha)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(args.planilha)
    elif ext == '.csv':
        df = pd.read_csv(args.planilha)
    else:
        sys.exit("Erro: formato de arquivo não suportado. Use xls, xlsx ou csv.")

    # Verifica a coluna de descrição
    if args.coluna not in df.columns:
        sys.exit(f"Erro: coluna '{args.coluna}' não encontrada na planilha")

    # Pré-processa e classifica
    df['_texto_limpo'] = df[args.coluna].apply(limpa_texto)
    df['_previsto'] = modelo.predict(df['_texto_limpo'])
    df['_previsto_label'] = df['_previsto'].map({1: 'TI', 0: 'NÃO TI'})

    # Determina arquivo de saída
    if args.saida:
        out_path = args.saida
    else:
        base, ext = os.path.splitext(args.planilha)
        out_path = f"{base}_classificado{ext}"

    # Salva
    if ext in ['.xls', '.xlsx']:
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Planilha classificada salva em: {out_path}")

if __name__ == "__main__":
    main()








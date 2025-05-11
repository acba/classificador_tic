#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import re

import pandas as pd
import joblib

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

def carrega_regex(path: str) -> list[str]:
    """
    Lê a planilha de filtros e retorna a lista de padrões regex.
    """
    df = pd.read_excel(path)

    # adiciona regex para termos que devem ser pesquisados com as palvras exatas
    df.loc[df.exato.notna(), ['termo']] = r'(?:\s|^)' + df['termo'] + r'(?:\s|$)'

    # remove acentuacao dos termo de TI
    df['termo'] = df['termo'].astype(str).str.normalize('NFKD')

    return df['termo'].dropna().astype(str).tolist()

def match_regex(texto: str, patterns: list[str]) -> bool:
    """
    Retorna True se qualquer pattern bater no texto (case-insensitive).
    """
    for pat in patterns:
        if re.search(pat, texto, flags=re.IGNORECASE):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Classifica descrições de uma planilha usando um modelo pré-treinado"
    )
    parser.add_argument(
        "--planilha",
        required=True,
        help="Caminho para o arquivo da planilha (xls, xlsx ou csv)"
    )
    parser.add_argument(
        "--coluna",
        required=True,
        help="Nome da coluna que contém o texto a ser classificado"
    )
    parser.add_argument(
        "--modelo",
        default="classificador_treinado.pkl",
        help="Caminho para o arquivo do modelo serializado (padrão: classificador_treinado.pkl)"
    )
    parser.add_argument(
        "--filtros",
        default="filtros/objeto.xlsx",
        help="Planilha com termos regex (padrão: filtros/objeto.xlsx)"
    )
    parser.add_argument(
        "--saida",
        help="Caminho de saída para salvar a planilha classificada (por padrão adiciona '_classificado' antes da extensão)"
    )
    args = parser.parse_args()

    # Verifica existência do modelo
    if not os.path.isfile(args.modelo):
        sys.exit(f"Erro: modelo não encontrado em '{args.modelo}'")
    if not os.path.isfile(args.filtros):
        sys.exit(f"Erro: planilha de filtros não encontrada em '{args.filtros}'")
    if not os.path.isfile(args.planilha):
        sys.exit(f"Erro: planilha de entrada não encontrada em '{args.planilha}'")

    # Carrega o modelo (pipeline completo)
    modelo = joblib.load(args.modelo)

    # Carrega os padrões regex
    termos = carrega_regex(args.filtros)

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
    df['_previsto_modelo'] = modelo.predict(df['_texto_limpo'])
    df['_previsto_modelo_label'] = df['_previsto_modelo'].map({1: 'TI', 0: 'NÃO TI'})

    # Predição por regex
    df['_previsto_regex'] = df['_texto_limpo'].apply(lambda t: int(match_regex(str(t), termos)))
    df['_previsto_regex_label'] = df['_previsto_regex'].map({1: 'TI', 0: 'NÃO TI'})

    df['_previsto_ensemble'] = df['_previsto_modelo'] + df['_previsto_regex']

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
    print("Colunas adicionadas:")
    print(" - _previsto_modelo_label : predição do modelo (TI / NÃO TI)")
    print(" - _previsto_regex_label  : predição por regex (TI / NÃO TI)")


if __name__ == "__main__":
    main()
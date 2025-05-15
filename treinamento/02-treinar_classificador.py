#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import joblib
import shutil
import argparse
import subprocess
from datetime import datetime
from unidecode import unidecode

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix
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

def limpa_texto(texto: str) -> str:
    """
    Normaliza, remove acentos, pontuação e excesso de espaços.
    """
    texto = texto.lower()
    # texto = unicodedata.normalize('NFKD', texto)
    texto = unidecode(texto)
    texto = re.sub(r'\W+', ' ', texto).replace('_', ' ')
    texto = re.sub(r'\s+', ' ', texto).strip()

    texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
    texto = ' '.join(texto.split())
    return texto

def carrega_regex(path: str) -> list[str]:
    """
    Lê a planilha de filtros e retorna a lista de padrões regex.
    """
    df = pd.read_excel(path)

    # adiciona regex para termos que devem ser pesquisados com as palavras exatas
    df.loc[df.exato.notna(), ['termo']] = r'(?:\s|^)' + df['termo'] + r'(?:\s|$)'

    # remove acentuacao dos termos
    df['termo'] = df['termo'].apply(unidecode)

    return df['termo'].dropna().astype(str).tolist()

def match_regex(texto: str, patterns: list[str]) -> bool:
    """
    Retorna True se qualquer pattern bater no texto (case-insensitive).
    """
    for pat in patterns:
        if re.search(pat, texto, flags=re.IGNORECASE):
            return True
    return False

def treina_e_testa(
        bd,
        balanceamento=True,
        classificador='regressaologistica',
        ngram=3,
        min_df=3,
        max_df=.8,
        filtro_regex=False,
        salva_log=True,
        salva_modelo=True,
        print_mode: str = 'both'
    ):
    """
    Função para treinar e testar o modelo.

    print_mode:
      - 'both'    -> prints no console e no log
      - 'console' -> somente no console
      - 'log'     -> somente no arquivo de log
      - 'none'    -> não imprime nada
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    TMP_PATH = 'tmp/'
    os.makedirs(TMP_PATH, exist_ok=True)

    OUTPUT_PATH = 'out/'
    modelos_path = f'{OUTPUT_PATH}modelos/'
    if salva_modelo:
        os.makedirs(modelos_path, exist_ok=True)

    original_stdout = sys.stdout
    log_file = None

    if salva_log and print_mode in ('both', 'log'):
        log_file = open(f"{TMP_PATH}log_treinamento.log", "a", encoding="utf-8")

     # redireciona saída de acordo com print_mode
    if print_mode == 'both' and log_file:
        sys.stdout = Tee(original_stdout, log_file)
    elif print_mode == 'log' and log_file:
        sys.stdout = log_file
    # sys.stdout = Tee(sys.stdout, f)

    if print_mode == 'none':
        sys.stdout = open(os.devnull, 'w')

    # Realiza balanceamento da base de dados, caso necessário
    bd_filepath = bd
    if balanceamento:
        bd_filepath = f"{TMP_PATH}{timestamp}_base_dados_balanceada.xlsx"
        subprocess.run(
                ["python", "01-balancear.py", "--planilha", f"{bd}", '--saida', f"{bd_filepath}"],
                check=True
            )
        print(f'Banco de dados {bd} foi balanceado')

    # Carrega a base de dados
    ext = os.path.splitext(bd_filepath)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(bd_filepath)
    elif ext == '.csv':
        df = pd.read_csv(bd_filepath)
    else:
        sys.exit("Erro: formato de arquivo não suportado. Use xls, xlsx ou csv.")

    if 'descricao' not in df.columns or 'classe' not in df.columns:
        sys.exit("Erro: colunas 'descricao' e 'classe' não encontradas na planilha")

    df = df[['descricao', 'classe'] + [col for col in df.columns if col not in ['descricao', 'classe']]]

    # df.columns = ['descricao', 'classe'] + list(df.columns[2:])
    # df['classe'].replace({'Sim': 1, 'Não': 0}, inplace=True)
    # df['classe'] = df['classe'].replace({'Sim': 1, 'Não': 0})
    df['classe'] = df['classe'].map({'Sim': 1, 'Não': 0}).astype(int)

    print("Base de dados original:", bd)
    print('Houve balanceamento da base de dados:', balanceamento)
    print("Base de dados balanceada:", bd_filepath)
    print()
    print(f'Para a base será considerado que a 1a coluna contém a descrição do objeto e a 2a coluna contém a classe')
    print(f'As classes serão mapeadas para 1 (TI) e 0 (Não TI), respectivamente')
    print("Tamanho da base de dados usada no treinamento:", len(df))
    print("Distribuição :", df['classe'].value_counts())
    print()

    # Remove linhas com valores nulos
    df = df.dropna(subset=['descricao', 'classe']).copy()
    df['texto_limpo'] = df['descricao'].apply(limpa_texto)

    print("=== Parâmetros de treinamento ===\n")

    tamanho_teste = 0.2
    tamanho_treino = 1 - tamanho_teste
    print(f"Tamanho do conjunto de treino: {tamanho_treino:.0%} - {tamanho_treino * len(df):,.0f} instâncias")
    print(f"Tamanho do conjunto de teste: {tamanho_teste:.0%} - {tamanho_teste * len(df):,.0f} instâncias")

    # Divisão treino/teste
    df_train, df_test = train_test_split(
        df,
        test_size=tamanho_teste,
        stratify=df['classe'],
        random_state=42
    )

    X_train = df_train['texto_limpo']
    y_train = df_train['classe']
    X_test  = df_test['texto_limpo']
    y_test  = df_test['classe']

    print('\nTF-IDF Vectorizer params:')
    # min_df = 3
    # max_df = 0.8
    ngram_range = (1, ngram)
    print(f'\nmin_df={min_df}, max_df={max_df}')
    print(f'ngram_range={ngram_range}')

    # 4) Pipeline TF–IDF + Logistic Regression
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)

    if classificador == 'regressaologistica':
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'))
        ])
        param_grid = {
            'clf__C': [10, 20, 30, 50, 60, 65, 70, 75, 80, 85, 90, 100, 1000],
            'clf__penalty': ['l1', 'l2']
        }
    elif classificador == 'svc':
        pipeline = Pipeline([
            ('tfidf', tfidf),
            # ('clf', SVC(verbose=True))
            ('clf', SVC(class_weight='balanced', verbose=True))
        ])
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10, 25, 50, 75, 100],
            'clf__kernel': ['linear'],
            # 'clf__kernel': ['linear', 'rbf'],
            # 'clf__gamma': ['scale', 'auto'],  # só usado quando kernel='rbf'
        }
    else:
        sys.exit("Erro: classificador não suportado. Use 'regressaologistica' ou 'svc'.")

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
    print("\nMelhor combinação de hiperparâmetros:", grid.best_params_)
    print(f"Melhor F1 (CV): {grid.best_score_:.4f}")

    # Avaliação no conjunto de teste com threshold default (0.5)
    modelo = grid.best_estimator_
    y_pred = modelo.predict(X_test)
    y_modelo = y_pred
    if filtro_regex:
        # Aplica regex para filtrar previsões
        termos_ti = carrega_regex(filtro_regex)
        y_pred_regex = X_test.apply(lambda t: int(match_regex(str(t), termos_ti)))
        y_pred = y_pred & y_pred_regex

        df_tmp = df_test
        df_tmp['previsto_reglog'] = y_modelo
        df_tmp['previsto_regex'] = y_pred_regex
        df_tmp['previsto_final'] = y_pred
        df_tmp['delta'] = df_tmp['previsto_final'] - df_tmp['classe']
        df_tmp.sort_values(by='delta').to_excel(f"teste_regex.xlsx", index=False)

    print("\n=== Resultados com conjunto de teste ===\n")
    print("\nRelatório de classificação ===")
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

    resultados = df_test[['descricao', 'texto_limpo', 'classe']].copy()
    resultados['previsto_modelo'] = y_pred
    resultados['esperado_label'] = resultados['classe'].map({1: 'TI', 0: 'NÃO TI'})
    resultados['previsto_modelo_label'] = resultados['previsto_modelo'].map({1: 'TI', 0: 'NÃO TI'})

    # Exibir 20 primeiros exemplos INCORRETOS
    incorretos = resultados[resultados['esperado_label'] != resultados['previsto_modelo_label']]
    print(f"\n--- {min(len(incorretos), 20)} primeiros exemplos onde previsto != esperado ---")
    print(
        incorretos[['descricao', 'esperado_label', 'previsto_modelo_label']]
        .head(20)
        .to_markdown(index=False, headers='keys', tablefmt='psql')
    )

    # Salva o modelo treinado completo
    if salva_modelo:
        balanceado = 'base_balanceada' if balanceamento else 'base_nao_balanceada'
        tecnica = 'regressaologistica' if classificador == 'regressaologistica' else 'svc'
        _filtro_regex = 'com_filtro' if filtro_regex else 'sem_filtro'

        base_modelo = f"{timestamp}_classificador_{balanceado}_{tecnica}_{_filtro_regex}_ngram_{ngram}_min_df_{min_df}"

        print(f"\nModelo salvo em: {modelos_path}")

        shutil.copy(bd_filepath, f"{modelos_path}{base_modelo}_bd.xlsx")
        if log_file:
            shutil.copy(f"{TMP_PATH}log_treinamento.log", f"{modelos_path}{base_modelo}_modelo.log")
        joblib.dump(modelo, f'{modelos_path}{base_modelo}_modelo.pkl')
        incorretos.to_excel(f"{modelos_path}{base_modelo}_erros_teste.xlsx", index=False)

    if log_file:
        sys.stdout = original_stdout
        log_file.flush()
        log_file.close()

    shutil.rmtree('tmp', ignore_errors=True)

    return grid.best_score_, grid.best_params_, tn, fp, fn, tp


def main():
    parser = argparse.ArgumentParser(
        description="Classifica descrições de uma planilha usando um modelo pré-treinado"
    )

    parser.add_argument(
        "--bd",
        required=True,
        default="base_dados.xlsx",
        help="Caminho para o arquivo da base de dados (treinamento e teste) (xls, xlsx ou csv) (padrão: base_dados.xlsx)"
    )
    parser.add_argument(
        "--balanceamento",
        default='y',
        help="Flag para indicar se deve ser feito o rebalanceamento entre as classes (padrão: y)"
    )
    parser.add_argument(
        "--salva_log",
        default='y',
        help="Flag para indicar se deve ser salvo o log em arquivo (padrão: y)"
    )
    parser.add_argument(
        "--salva_modelo",
        default='y',
        help="Flag para indicar se deve ser salvo o modelo em arquivo (padrão: y)"
    )
    parser.add_argument(
        "--filtro_ti_regex",
        default=None,
        help="Indica a planilha contendo os termos de TI a ser aplicado ao final do modelo (padrão: None)"
    )
    parser.add_argument(
        "--classificador",
        choices=['regressaologistica', 'svc'],
        default='regressaologistica',
        help="Indique a técnica utilizada para realizar a classificação do modelo (padrão: regressaologistica)"
    )
    parser.add_argument(
        "--ngram",
        default='3',
        help="ngram para o TF-IDF (padrão: 3)"
    )
    parser.add_argument(
        "--min_df",
        default='1',
        help="min_df para o TF-IDF (padrão: 1)"
    )
    parser.add_argument(
        "--saida",
        help="Caminho de saída para salvar o modelo"
    )
    args = parser.parse_args()

    salva_log = args.salva_log.lower() == 'y'
    balanceamento = args.balanceamento.lower() == 'y'
    salva_modelo = args.salva_modelo.lower() == 'y'
    ngram = int(args.ngram)
    min_df = int(args.min_df)

    f1, bp, tn, fp, fn, tp = treina_e_testa(args.bd, balanceamento, args.classificador, ngram, min_df, .8, args.filtro_ti_regex, salva_log, salva_modelo)


if __name__ == "__main__":
    main()
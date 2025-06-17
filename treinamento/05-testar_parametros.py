#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib.util
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
        description="Testa os parâmetros de modelos diferentes em busca do melhor classificador"
    )

parser.add_argument(
    "--bd",
    required=True,
    default="base_dados_original_revisado.xlsx",
    help="Caminho para o arquivo da base de dados (treinamento e teste) (xls, xlsx ou csv) (padrão: base_dados.xlsx)"
)

args = parser.parse_args()

# 1. Cria um spec para o arquivo
spec = importlib.util.spec_from_file_location(
    "mod_treino",          # nome arbitrário pra esse módulo em tempo de execução
    "02-treinar_classificador.py"
)

# 2. Cria o módulo a partir desse spec
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
treina_e_testa = mod.treina_e_testa

# 2. Nome do arquivo de resultados
RESULT_FILE = 'resultados_treino.xlsx'

# 3. Se já existir, lê; senão, cria DataFrame vazio com colunas
if os.path.exists(RESULT_FILE):
    df_res = pd.read_excel(RESULT_FILE)
else:
    df_res = pd.DataFrame(
        columns=['balancear','classificador','ngram','min_df','max_df','f1','tn','fp','fn','tp', 'filtro_regex', 'bp']
    )

# 4. Cria um set de tuplas (parâmetros) já executados
executados = {
    (row.balancear, row.classificador, row.filtro_regex, row.ngram, row.min_df, row.max_df)
    for row in df_res.itertuples()
}

# 5. Liste aqui os seus grids de parâmetros
param_grid = {
    'balancear': [True, False],
    'classificador': ['regressaologistica'],    
    # 'classificador': ['svc'],
    # 'classificador': ['regressaologistica', 'svc'],
    'filtro_regex': [False],
    # 'filtro_regex': ['../filtros/objeto.xlsx', False],
    'ngram': [2, 3, 4],
    'min_df': [1, 2, 3],
    'max_df': [0.8]
}

# 6. Gera todas as combinações
from itertools import product
combos = product(
    param_grid['balancear'],
    param_grid['classificador'],
    param_grid['filtro_regex'],
    param_grid['ngram'],
    param_grid['min_df'],
    param_grid['max_df'],
)

# 7. Loop principal
for balancear, classificador, filtro_regex, ngram, min_df, max_df in combos:
    chave = (balancear, classificador, filtro_regex, ngram, min_df, max_df)
    if chave in executados:
        print(f"Já executado: balancear={balancear}, classificador={classificador}, filtro_regex={filtro_regex}, "
              f"ngram={ngram}, min_df={min_df}, max_df={max_df} — pulando.")
        continue

    print(f"Executando: balancear={balancear}, classificador={classificador}, filtro_regex={filtro_regex}, "
          f"ngram={ngram}, min_df={min_df}, max_df={max_df}")

    # 8. Chama a função
    f1, bp, tn, fp, fn, tp = treina_e_testa(
        bd=args.bd,
        balanceamento=balancear,
        classificador=classificador,
        ngram=ngram,
        min_df=min_df,
        max_df=max_df,
        filtro_regex=filtro_regex,
        salva_log=True,
        salva_modelo=True,
        print_mode='both'
    )

    # 9. Monta o registro e adiciona ao DataFrame
    nova_linha = {
        'balancear': balancear,
        'classificador': classificador,
        'filtro_regex': filtro_regex,
        'ngram': ngram,
        'min_df': min_df,
        'max_df': max_df,
        'f1': f1,
        'bp': bp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    df_res = pd.concat([df_res, pd.DataFrame([nova_linha])], ignore_index=True)

    # 10. Atualiza set de executados e grava o arquivo
    executados.add(chave)
    df_res.to_excel(RESULT_FILE, index=False)
    print(f"✔ Registro salvo: {RESULT_FILE}\n")

print("Todas as combinações processadas.")

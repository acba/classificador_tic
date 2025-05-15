import os
import sys
import argparse

import pandas as pd
import numpy as np

from sklearn.utils import resample

def main():
    parser = argparse.ArgumentParser(
        description="Classifica descrições de uma planilha usando um modelo pré-treinado"
    )
    parser.add_argument(
        "--planilha",
        required=True,
        default="base_dados.xlsx",
        help="Caminho para o arquivo da planilha (xls, xlsx ou csv)"
    )
    parser.add_argument(
        "--col_descricao",
        default="descricao",
        help="Coluna que contém a descrição (padrão: descricao)"
    )
    parser.add_argument(
        "--col_classe",
        default="classe",
        help="Coluna que contém a classe (padrão: classe)"
    )
    parser.add_argument(
        "--debug",
        default="n",
        help="Flag para ativar o debug (padrão: n)"
    )
    parser.add_argument(
        "--saida",
        help="Caminho de saída para salvar a base rebalanceada (por padrão adiciona o sufixo '_balanceado')"
    )

    args = parser.parse_args()

    # Carrega a planilha
    ext = os.path.splitext(args.planilha)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(args.planilha)
    elif ext == '.csv':
        df = pd.read_csv(args.planilha)
    else:
        sys.exit("Erro: formato de arquivo não suportado. Use xls, xlsx ou csv.")

    if args.col_descricao not in df.columns:
        sys.exit(f"Erro: coluna '{args.col_descricao}' não encontrada na planilha")
    if args.col_classe not in df.columns:
        sys.exit(f"Erro: coluna '{args.col_classe}' não encontrada na planilha")

    df.rename(columns={args.col_classe: 'classe', args.col_descricao: 'descricao'}, inplace=True)

    df = df[['descricao', 'classe'] + [col for col in df.columns if col not in ['descricao', 'classe']]]

    # --- Verifica balanceamento das classes ---
    contagem = df['classe'].value_counts()
    if args.debug == 'y':
        print("Contagem por classe antes do balanceamento:\n", contagem)

    # Se o desbalanceamento for grande (por exemplo, ratio > 1.2), faz oversampling
    ratio = contagem.max() / contagem.min()
    if ratio > 1.2:
        # Identifica classes majoritária e minoritária
        classe_majoritaria = contagem.idxmax()
        classe_minoritaria = contagem.idxmin()

        df_major = df[df['classe'] == classe_majoritaria]
        df_minor = df[df['classe'] == classe_minoritaria]

        # Oversampling da minoritária para igualar a majoritária
        df_minor_upsampled = resample(
            df_minor,
            replace=True,
            n_samples=len(df_major),
            random_state=42
        )

        # Reconstrói o DataFrame balanceado
        df = pd.concat([df_major, df_minor_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Determina arquivo de saída
        if args.saida:
            base, ext = os.path.splitext(args.saida)
            out_path = base
        else:
            base, ext = os.path.splitext(args.planilha)
            os.makedirs('out/', exist_ok=True)
            out_path = f"out/{base}_balanceado"

        df.to_excel(f'{out_path}.xlsx', index=False)
        if args.debug == 'y':
            print(f"Salvando base balanceada em: {out_path}")
            print(f"\nApós oversampling, contagem por classe:\n{df['classe'].value_counts()}")
    else:
        if args.debug == 'y':
            print("As classes já estão razoavelmente balanceadas; sem ajuste aplicado.")

if __name__ == "__main__":
    main()
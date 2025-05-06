import pandas as pd
import numpy as np
from sklearn.utils import resample

filename = 'base_dados'

df = pd.read_excel(f'{filename}.xlsx')
df = df[['descricao', 'TIC', 'Tipo']]

# --- Verifica balanceamento das classes ---
contagem = df['TIC'].value_counts()
print("Contagem por classe antes do balanceamento:\n", contagem)

# Se o desbalanceamento for grande (por exemplo, ratio > 1.2), faz oversampling
ratio = contagem.max() / contagem.min()
if ratio > 1.2:
    # Identifica classes majoritária e minoritária
    classe_majoritaria = contagem.idxmax()
    classe_minoritaria = contagem.idxmin()

    df_major = df[df['TIC'] == classe_majoritaria]
    df_minor = df[df['TIC'] == classe_minoritaria]

    # Oversampling da minoritária para igualar a majoritária
    df_minor_upsampled = resample(
        df_minor,
        replace=True,
        n_samples=len(df_major),
        random_state=42
    )

    # Reconstrói o DataFrame balanceado
    df = pd.concat([df_major, df_minor_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_excel(f'{filename}_balanceado.xlsx', index=False)
    print(f"\nApós oversampling, contagem por classe:\n{df['TIC'].value_counts()}")
else:
    print("As classes já estão razoavelmente balanceadas; sem ajuste aplicado.")

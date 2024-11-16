import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score
import numpy as np
import seaborn as sns

df = pd.read_csv("distorted-images/distorted-images-data.csv", sep=";")
df = df.rename(columns={"ORIG_CLASSES": "CLASS", "TOP_1_ACCURACYMEAN_CONFIDENCE_DROP": "TOP1_ACCURACY"})

if "CLASS" not in df.columns or "TOP1_ACCURACY" not in df.columns:
    print("Erro: As colunas 'CLASS' e 'TOP1_ACCURACY' não estão no DataFrame.")
else:
    mean_accuracy = df["TOP1_ACCURACY"].mean()
    print(f"Média da precisão de todos os resultados: {mean_accuracy:.2f}")

    true_labels = df["CLASS"].apply(lambda x: eval(x, {"np": np})[0][0])
    predicted_labels = df["TOP1_ACCURACY"].apply(lambda x: '1' if x > 0.5 else '0')

    print(f"True labels: {true_labels.unique()}")
    print(f"Predicted labels: {predicted_labels.unique()}")

    mean_f1_score = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)
    print(f"Média da F1-Score de todas as comparações: {mean_f1_score:.2f}")

    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    print(f"Precisão de todas as comparações: {precision:.2f}")

    accuracy_by_class = df.groupby("CLASS")["TOP1_ACCURACY"].mean()

    max_accuracy_class = accuracy_by_class.idxmax()
    min_accuracy_class = accuracy_by_class.idxmin()

    print(f"Classe com maior precisão: {max_accuracy_class} ({accuracy_by_class[max_accuracy_class]:.2f})")
    print(f"Classe com menor precisão: {min_accuracy_class} ({accuracy_by_class[min_accuracy_class]:.2f})")

    # Gráfico de Barras
    plt.figure(figsize=(10, 5))
    accuracy_by_class.plot(kind='bar', color='skyblue')
    plt.title('Precisão Média por Classe')
    plt.ylabel('Precisão Média')
    plt.gca().axes.get_xaxis().set_visible(False)  # Oculta o eixo X
    plt.tight_layout()
    plt.show()

    # Gráfico de Violino
    df["CLASS"] = df["CLASS"].apply(lambda x: eval(x, {"np": np})[0][0])
    df["TOP1_ACCURACY"] = df["TOP1_ACCURACY"].astype(float)

    violin_data = [df[df["CLASS"] == cls]["TOP1_ACCURACY"].dropna() for cls in df["CLASS"].unique()]

    plt.figure(figsize=(14, 7))
    plt.violinplot(violin_data, showmeans=True)
    plt.title('Distribuição da Precisão por Classe (Violin Plot)', fontsize=16)
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Precisão', fontsize=14)
    plt.xticks(ticks=range(1, len(df["CLASS"].unique()) + 1), labels=df["CLASS"].unique(), rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()

    # Gráfico de Densidade de Kernel para SSIM Gaussiano vs Precisão
    plt.figure(figsize=(10, 5))
    kde = sns.kdeplot(x=df["GAUSS_SSIM"], y=df["TOP1_ACCURACY"], fill=True, cmap="Blues")
    plt.title('Densidade de Kernel: SSIM Gaussiano vs Precisão')
    plt.xlabel('SSIM Gaussiano')
    plt.ylabel('Precisão')
    plt.colorbar(kde.collections[0], label='Densidade')
    plt.tight_layout()
    plt.show()

    # Gráfico de Densidade de Kernel para SSIM Blur vs Precisão
    plt.figure(figsize=(10, 5))
    kde = sns.kdeplot(x=df["BLUR_SSIM"], y=df["TOP1_ACCURACY"], fill=True, cmap="Greens")
    plt.title('Densidade de Kernel: SSIM Blur vs Precisão')
    plt.xlabel('SSIM Blur')
    plt.ylabel('Precisão')
    plt.colorbar(kde.collections[0], label='Densidade')
    plt.tight_layout()
    plt.show()

    # Gráfico de Densidade de Kernel para SSIM Grayscale vs Precisão
    plt.figure(figsize=(10, 5))
    kde = sns.kdeplot(x=df["GRAYSCALE_SSIM"], y=df["TOP1_ACCURACY"], fill=True, cmap="Reds")
    plt.title('Densidade de Kernel: SSIM Grayscale vs Precisão')
    plt.xlabel('SSIM Grayscale')
    plt.ylabel('Precisão')
    plt.colorbar(kde.collections[0], label='Densidade')
    plt.tight_layout()
    plt.show()
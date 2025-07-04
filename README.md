# 🖼️ Image Distortion Analysis with Deep Learning

Um projeto de análise de modelos de deep learning para classificação de imagens sob diferentes tipos de distorções.


## 📋 Descrição


Este projeto avalia como diferentes tipos de distorções de imagem afetam a performance de modelos de classificação baseados em ResNet50. O sistema aplica várias transformações nas imagens e analisa o impacto na precisão das predições usando métricas como SSIM (Structural Similarity Index Measure) e Top-1 Accuracy.

## 🎯 Objetivos

- **Análise de Robustez**: Avaliar como modelos de deep learning se comportam com imagens distorcidas

- **Comparação de Distorções**: Comparar o impacto de diferentes tipos de ruído e transformações

- **Visualização de Dados**: Gerar gráficos informativos para análise dos resultados

- **Métricas de Qualidade**: Calcular SSIM para medir a similaridade estrutural entre imagens

## 🛠️ Tecnologias Utilizadas

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)





- **TensorFlow/Keras**: Modelo ResNet50 pré-treinado


- **OpenCV**: Processamento de imagens


- **Scikit-image**: Cálculo de métricas SSIM


- **Pandas**: Manipulação de dados


- **Matplotlib/Seaborn**: Visualização de dados


- **Scikit-learn**: Métricas de avaliação



🚀 Como Usar


1. Preparação dos Dados

Organize suas imagens nos diretórios train/ e val/ seguindo a estrutura:

```bash


train/n01440764/

val/n01440764/


```





2. Executar o Pipeline


```bash

python Pipeline_distortion.py

```


4. Análise Interativa


Abra o notebook test_run.ipynb para análises detalhadas.





📊 Tipos de Distorções Analisadas


| Distorção               | Descrição                | Parâmetros        |


| ----------------------- | ------------------------ | ----------------- |


| 🌫️ **Ruído Gaussiano** | Adiciona ruído aleatório | `mean=10, std=10` |


| 🌀 **Blur Gaussiano**   | Desfoque da imagem       | `kernel=(11,11)`  |


| ⚫ **Escala de Cinza**   | Conversão para grayscale | -                 |


| 🔄 **Efeito Negativo**  | Inversão de cores        | -                 |


| 🔍 **Zoom**             | Ampliação da imagem      | `fator=1.5`       |



Métricas Calculadas




    SSIM (Structural Similarity Index): Mede a similaridade estrutural entre imagens.

    Top-1 Accuracy: Precisão da classificação no melhor resultado.

    F1-Score: Métrica balanceada que combina precisão e recall.

    Precision: Precisão das predições feitas pelo modelo.



Tipos de Gráficos



    📊 Gráfico de Barras: Exibe a precisão média por classe.

    🎻 Violin Plot: Mostra a distribuição da precisão por classe.

    🌈 Density Plot: Representa a relação entre SSIM e precisão usando um mapa de calor.



Interpretação dos Gráficos de Densidade

    Cores mais escuras: Maior concentração de dados.

    Cores mais claras: Menor concentração de dados.

    Barra de cores: Indica a densidade relativa dos pontos no gráfico.





🔬 Funcionalidades Principais
classifier.py

    Classificação com ResNet50

    Cálculo de SSIM entre imagens

    Funções de processamento de imagem

image_distortion.py

    Aplicação de ruído gaussiano

    Blur e efeitos de distorção

    Transformações geométricas

graficos.py

    Gráficos de barras limpos (sem eixo X)

    Violin plots para distribuição

    Mapas de densidade com legenda

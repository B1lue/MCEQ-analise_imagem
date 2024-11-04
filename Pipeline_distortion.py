import os
import pandas as pd
from classifier import (classify_to_data, open_img, ssim, top_1_accuracy)
from image_distortion import (
    add_gaussian_noise, apply_gaussian_blur, convert_to_grayscale,
    apply_negative_effect, zoom_image
)

PATH_TO_DIR_VAL = "val/n01440764/"
PATH_TO_DIR_TRAIN = "train/n01440764/"
PATH_TO_SAVE = "distorted-images/"

HEADERS = [
    "FILE_PATH",
    "ORIG_CLASSES",
    "GAUSS_SSIM",
    "GAUSS_CLASSES",
    "BLUR_SSIM",
    "BLUR_CLASSES",
    "GRAYSCALE_SSIM",
    "GRAYSCALE_CLASSES",
    "NEGATIVE_SSIM",
    "NEGATIVE_CLASSES",
    "ZOOM_SSIM",
    "ZOOM_CLASSES",
    "TOP_1_ACCURACY"
    "MEAN_CONFIDENCE_DROP"
]
df = pd.DataFrame(columns=HEADERS)

# Cria o diretório se ele não existir
try:
    os.mkdir(PATH_TO_SAVE)
except FileExistsError:
    pass

# Função que processa os arquivos de um diretório
def process_files(files, path_to_dir):
    global df  # Permite a modificação da variável global

    # Itera sobre os arquivos
    for file in files:
        path = path_to_dir + file
        save = PATH_TO_SAVE + file.removesuffix(".JPEG")

        orig_img = open_img(path=path)
        classes_orig, top_class_orig = classify_to_data(orig_img)

        # Aplica ruído Gaussiano
        after_gauss = add_gaussian_noise(orig_img, mean=10, std=10)
        ssim_gauss = ssim(orig_img, after_gauss)
        classes_gauss, top_class_gauss = classify_to_data(after_gauss)

        # Aplica desfoque Gaussiano
        after_blur = apply_gaussian_blur(orig_img, (11, 11), 0)
        ssim_blur = ssim(orig_img, after_blur)
        classes_blur, top_class_blur = classify_to_data(after_blur)

        # Converte para escala de cinza
        after_grayscale = convert_to_grayscale(orig_img)
        ssim_grayscale = ssim(orig_img, after_grayscale)
        classes_grayscale, top_class_grayscale = classify_to_data(after_grayscale)

        # Aplica efeito negativo
        after_negative = apply_negative_effect(orig_img)
        ssim_negative = ssim(orig_img, after_negative)
        classes_negative, top_class_negative = classify_to_data(after_negative)

        # Aplica zoom
        after_zoom = zoom_image(orig_img, 1.5)
        ssim_zoom = ssim(orig_img, after_zoom)
        classes_zoom, top_class_zoom = classify_to_data(after_zoom)

        # Calcula a Top-1 Accuracy
        true_labels = [top_class_orig] * 5
        predicted_labels = [top_class_gauss, top_class_blur, top_class_grayscale, top_class_negative, top_class_zoom]
        top_1_acc = top_1_accuracy(true_labels, predicted_labels)



        # Adiciona os resultados ao DataFrame
        df.loc[-1] = [
            path,
            classes_orig,
            ssim_gauss,
            classes_gauss,
            ssim_blur,
            classes_blur,
            ssim_grayscale,
            classes_grayscale,
            ssim_negative,
            classes_negative,
            ssim_zoom,
            classes_zoom,
            top_1_acc
        ]
        df.index = df.index + 1
        df = df.sort_index()

# Verifica se os diretórios existem antes de processar os arquivos
if os.path.exists(PATH_TO_DIR_VAL):
    files_val = os.listdir(PATH_TO_DIR_VAL)
    process_files(files_val, PATH_TO_DIR_VAL)
else:
    print(f"O diretório {PATH_TO_DIR_VAL} não foi encontrado.")

if os.path.exists(PATH_TO_DIR_TRAIN):
    files_train = os.listdir(PATH_TO_DIR_TRAIN)
    process_files(files_train, PATH_TO_DIR_TRAIN)
else:
    print(f"O diretório {PATH_TO_DIR_TRAIN} não foi encontrado.")


df.to_csv("distorted-images/distorted-images-data.csv", sep=";")
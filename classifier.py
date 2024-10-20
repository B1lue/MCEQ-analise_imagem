from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import cv2
from skimage.metrics import structural_similarity

MODEL = ResNet50(weights='imagenet')

# Função que classifica uma imagem
def classify(img):
    # Redimensiona a imagem para 224x224 pixels
    try:
        x = cv2.resize(img, (224, 224))
        x = x[:, :, ::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds)[0]

        # Exibe as classes e as probabilidades
        for c in classes:
            print("\t%s (%s): %.2f%%" % (c[1], c[0], c[2] * 100))
        return classes
    # Exceção caso a classificação falhe
    except Exception as e:
        print("Classification failed:", e)
        return []

# Função que converte uma imagem para dados
def classify_to_data(img):
    classes = classify(img)
    return [(c[1], c[2]) for c in classes]

# Função que abre uma imagem
def open_img(path):
    return cv2.imread(path)

# Função que adiciona ruído gaussiano a uma imagem
def ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2) * 100

# Função que aplica ruído gaussiano a uma imagem
def jpeg(img, quality):
    _, x = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(x, cv2.IMREAD_COLOR)

# Função que redimensiona uma imagem
def resize(img, w, h):
    orig_h, orig_w = img.shape[:2]
    x = cv2.resize(img, (w, h))
    return cv2.resize(x, (orig_w, orig_h))

# Função que aplica o filtro de Canny em uma imagem
def canny(img):
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Canny(x, 100, 200)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
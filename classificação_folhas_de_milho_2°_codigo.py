!pip install mahotas

from google.colab import drive
import os
import numpy as np
import cv2
import mahotas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import xgboost as xgb

# Montagem do Google Drive
drive.mount('/content/drive')

# Definindo os caminhos das pastas
base_path_train = '/content/drive/MyDrive/folhas/train'
base_path_test = '/content/drive/MyDrive/folhas/test'
base_path_val = '/content/drive/MyDrive/folhas/val'

# Função para extrair características usando Haralick
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick_features = mahotas.features.haralick(gray).mean(axis=0)
    return haralick_features

def load_images_from_folder(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            features.append(extract_features(img))
            labels.append(label)
    return features, labels

# Carregar imagens das pastas treino, teste e validação
def load_dataset(base_path):
    features = []
    labels = []
    label_map = {
        'Murcha_da_folha_do_norte': 0,
        'Mancha_foliar_de_Cercospora': 1,
        'Ferrugem_comum': 2,
        'Milho_saudavel': 3
    }
    for label_name, label in label_map.items():
        folder = os.path.join(base_path, label_name.strip())
        f, l = load_images_from_folder(folder, label)
        features.extend(f)
        labels.extend(l)
    return np.array(features), np.array(labels)

# Carregar conjuntos de dados
X_train, y_train = load_dataset(base_path_train)
X_test, y_test = load_dataset(base_path_test)
X_val, y_val = load_dataset(base_path_val)

# Treinamento do modelo XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'num_class': 4,
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.8
}

evals = [(dtrain, 'train'), (dtest, 'eval'), (dval, 'val')]
results = {}
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, evals_result=results, verbose_eval=True)

y_pred = model.predict(dtest)
predictions = [int(value) for value in y_pred]

# Métricas de desempenho
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Matriz de confusão e relatório de classificação
labels = ['Murcha da Folha do Norte', 'Mancha Foliar de Cercospora', 'Ferrugem Comum', 'Milho Saudável']
cm = confusion_matrix(y_test, predictions)
cr = classification_report(y_test, predictions, target_names=labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap=plt.cm.Blues)

plt.title('Matriz de Confusão')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.xticks(rotation=45)
plt.show()
print(cr)

# Plotando a perda (logloss) do treino, teste e validação ao longo do tempo
epochs = len(results['train']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['train']['mlogloss'], label='Treino')
plt.plot(x_axis, results['eval']['mlogloss'], label='Teste')
plt.plot(x_axis, results['val']['mlogloss'], label='Validação')
plt.xlabel('Número de Épocas')
plt.ylabel('Log Loss')
plt.title('Treino, Teste e Validação - XGBoost')
plt.legend()
plt.show()
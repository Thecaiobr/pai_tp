import os
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def mahalanobis_distance(x, mean, inv_covariance_matrix):
    x_minus_mean = x - mean
    return np.sqrt(np.dot(np.dot(x_minus_mean, inv_covariance_matrix), x_minus_mean.T))

def predict_mahalanobis(data, multiclass=False):
    classes = data["bethesda_system"].unique()

    class_covariance_matrices = {}
    class_means = {}
    for c in classes:
        class_data = data[data["bethesda_system"] == c].drop(["bethesda_system", "original_filename", "cell_id"], axis=1)
        covariance_matrix = LedoitWolf().fit(class_data).covariance_
        class_covariance_matrices[c] = np.linalg.inv(covariance_matrix)
        class_means[c] = np.mean(class_data, axis=0)

    predicted_labels = []
    for _, sample in data.iterrows():
        distances = {c: mahalanobis_distance(sample[3:], class_means[c], class_covariance_matrices[c]) for c in classes}
        predicted_label = min(distances, key=distances.get)
        predicted_labels.append(predicted_label)

    # Avaliação da acurácia
    accuracy = accuracy_score(data["bethesda_system"], predicted_labels)
    print(f'Acurácia: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(data["bethesda_system"], predicted_labels, labels=classes)
    df = pd.DataFrame(conf_matrix, index=classes, columns=classes)

    # Criação do diretório se não existir
    os.makedirs('confusion_matriz_mahalanobis', exist_ok=True)

    if multiclass:
        df.to_csv('confusion_matriz_mahalanobis/matriz_confusao_multiclass.csv', index=False)
    else:
        df.to_csv('confusion_matriz_mahalanobis/matriz_confusao_binary.csv', index=False)

def main():
    characteristics_df = pd.read_csv('characteristics.csv')
    train, test = train_test_split(characteristics_df, test_size=0.2, random_state=45, shuffle=True, stratify=characteristics_df['bethesda_system'])

    print("Classificação Binária:")
    predict_mahalanobis(train)
    
    print("\nClassificação Multiclasse:")
    predict_mahalanobis(test, multiclass=True)

if __name__ == "__main__":
    main()

import numpy as np
from scipy.spatial.distance import mahalanobis


#Classe negativa e nçao negativa, substituir dados logico
class_A_data = np.array([[1, 2], [2, 1], [1.5, 1.5], [2, 2.5], [1.2, 1.8]])
class_B_data = np.array([[4, 5], [5, 4], [4.5, 4.5], [3.5, 4.2], [4.8, 3.7]])


mean_A = np.mean(class_A_data, axis=0)
cov_A = np.cov(class_A_data, rowvar=False)
inv_cov_A = np.linalg.inv(cov_A)

mean_B = np.mean(class_B_data, axis=0)
cov_B = np.cov(class_B_data, rowvar=False)
inv_cov_B = np.linalg.inv(cov_B)

predictions = [] # Data_test são os dados de treinamento em coordenadas
for point in data_test:
    distance_A = mahalanobis(point, mean_A, inv_cov_A)
    distance_B = mahalanobis(point, mean_B, inv_cov_B)
    predicted_class = "A" if distance_A < distance_B else "B"
    predictions.append(predicted_class)

# Calculando acurácia
correct_predictions = sum(pred == true for pred, true in zip(predictions, labels_test))
accuracy = correct_predictions / len(data_test)
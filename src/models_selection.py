import json
import matplotlib.pyplot as plt

# coppia: modello - numero di file
with open('./model-num_files.json', mode="r") as f:
    data = json.load(f)

# Estrai i modelli e il numero di file che li usano
num_files = list(x for x in data.values() if x >= 100)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Calcolo delle metriche statistiche
mean_value = np.mean(num_files)
median_value = np.median(num_files)
std_dev = np.std(num_files, ddof=1)
skewness = stats.skew(num_files)
kurtosis = stats.kurtosis(num_files)

# Restituzione dei risultati
print(f"Metriche statistiche per il numero di file per modello:")
print(f"Media: {mean_value:.2f}")
print(f"Mediana: {median_value:.2f}")
print(f"Deviazione Standard: {std_dev:.2f}")
print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurtosis:.2f}")

# Creazione del boxplot per visualizzare la distribuzione
plt.figure(figsize=(8, 5))
plt.boxplot(num_files, vert=False, patch_artist=True)
plt.title("Boxplot della Distribuzione del Numero di File per Modello")
plt.xlabel("Numero di File")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()


# Crea un istogramma per la distribuzione del numero di file
plt.figure(figsize=(10, 5))
plt.hist(num_files, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Numero di file che usano il modello')
plt.ylabel('Frequenza')
plt.title('Distribuzione del numero di file per modello')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# models_name = list(model for model, num in data.items() if num >= 100)
# tot_files = sum(num for num in data.values() if num >= 100)
# #print(tot_files)

# sanitized_models_name = [model_name.replace("/", "_") for model_name in models_name]

# print(len(sanitized_models_name))

# with open('models_name.json', 'w') as f:
#     json.dump(sanitized_models_name, f, indent=4)
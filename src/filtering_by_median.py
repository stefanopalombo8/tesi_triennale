import os
import numpy as np
import json
import concurrent.futures

def count_lines(file_path):
    """Conta il numero di righe in un file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def process_model(model):
    """Elabora una cartella (modello) e filtra i file in base alla lunghezza."""
    folder_path = os.path.join(base_folder_path, model)
    # Genera la lista dei file .py nella cartella corrente
    file_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".py")
    ]

    if not file_paths:
        print(f"{model}: nessun file .py trovato.")
        return model, []  # Restituisci una lista vuota se non ci sono file
    
    # Conta le righe di ogni file in parallelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        file_lengths = list(executor.map(count_lines, file_paths))
    
    # Calcola la mediana e l'intervallo interquartile (IQR)
    median = np.median(file_lengths)
    q1 = np.percentile(file_lengths, 25)
    q3 = np.percentile(file_lengths, 75)
    iqr = q3 - q1

    # Intervallo attorno alla mediana
    min_length = max(int(median - 0.25 * iqr), 1)
    max_length = int(median + 0.25 * iqr)
    
    # Seleziona i file che rientrano nell'intervallo stabilito
    filtered_files = [
        file_paths[i] for i in range(len(file_lengths))
        if min_length <= file_lengths[i] <= max_length
    ]
    
    print(f"{model}: {len(filtered_files)} file selezionati ({min_length}-{max_length} righe)")
    return model, filtered_files

# Carica la lista dei modelli da processare
with open("./models_name.json", "r") as f:
    models = json.load(f)

base_folder_path = r"C:\Users\fafup\Desktop\tirocinio_tesi_triennale\models_files"
output_data = {}

# Processa le cartelle in parallelo
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(process_model, models[250:500])

# Raccoglie i risultati
for model, filtered_files in results:
    output_data[model] = filtered_files

# Salva i risultati in un file JSON
output_path = os.path.join("./", "files_median.json")
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"\nRisultati salvati in: {output_path}")
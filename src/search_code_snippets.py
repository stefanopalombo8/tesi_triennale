import os
import json
import ast
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import concurrent.futures

def merge_intervals(intervals):
    """
    Data una lista di tuple (start, end), restituisce una lista di intervalli unificati,
    ordinati e senza sovrapposizioni.
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

# Funzione per estrarre snippet di codice rilevanti da un file Python
def extract_relevant_ast_with_context(file_content, keywords, context=2):
    """
    Usa l'AST per estrarre porzioni di codice (import e definizioni di funzione)
    che contengono le keyword (sia nel nome che nel corpo) e aggiunge 2 righe
    di contesto prima e dopo il nodo.

    Per evitare ripetizioni, raccoglie gli intervalli (start, end) delle righe e li unisce. 
    Perché altrimenti porzioni di codice vicine verrebbero considerate come snippet separati 
    ripetendosi.
    
    :param file_content: contenuto del file Python come stringa.
    :param keywords: lista di keyword da cercare.
    :param context: numero di righe di contesto da includere.
    :return: lista di snippet (ciascuno come stringa).
    """
    try:
        tree = ast.parse(file_content)
    except Exception as e:
        print("Errore nel parsing dell'AST:", e)
        return []
    
    lines = file_content.splitlines()
    n_lines = len(lines)
    intervals = []  # Lista di tuple (start, end) per ciascun blocco rilevante

    # Keyword specifiche per filtrare gli import
    import_filter_keywords = [
        "transformers", "huggingface", "automodel", "autotokenizer", 
        "autoconfig", "pipeline", "sklearn", "scikit-learn", "sentence-transformers"
    ]
    
    for node in ast.walk(tree):
        # Filtraggio per i nodi di importazione
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names_to_check = []
            if isinstance(node, ast.Import):
                names_to_check = [alias.name for alias in node.names if alias.name]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names_to_check.append(node.module)
                names_to_check.extend([alias.name for alias in node.names if alias.name])
            
            # Verifica se almeno uno dei nomi contiene una keyword d'interesse
            if any(any(filter_kw in name.lower() for filter_kw in import_filter_keywords) 
                   for name in names_to_check):
                start = node.lineno - 1  # indice base zero
                if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                    end = node.end_lineno
                else:
                    end = node.lineno
                start = max(0, start - context)
                end = min(n_lines, end + context)
                intervals.append((start, end))
                
        # Filtraggio per le definizioni di funzione
        elif isinstance(node, ast.FunctionDef):
            function_source = ast.get_source_segment(file_content, node)
            if function_source and (
                any(kw.lower() in node.name.lower() for kw in keywords) or 
                any(kw.lower() in function_source.lower() for kw in keywords)
            ):
                start = node.lineno - 1
                if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                    end = node.end_lineno
                else:
                    end = node.lineno
                start = max(0, start - context)
                end = min(n_lines, end + context)
                intervals.append((start, end))
    
    merged_intervals = merge_intervals(intervals)
    extracted_parts = []
    for start, end in merged_intervals:
        snippet = "\n".join(lines[start:end])
        extracted_parts.append(snippet)
    
    return extracted_parts

# Funzioni per la selezione dei "migliori" snippet
def select_best_snippets_by_clustering(snippets, embedding_model, num_clusters=3):
    """
    Seleziona i migliori snippet applicando un clustering sui loro embedding e 
    scegliendo, per ogni cluster, lo snippet più vicino al centroide.
    
    :param snippets: lista di snippet (stringhe)
    :param embedding_model: modello SentenceTransformer già caricato per ottenere gli embedding.
    :param num_clusters: numero di cluster da formare.
    :return: lista di snippet rappresentativi per cluster.
    """
    if not snippets:
        return []
    
    embeddings = embedding_model.encode(snippets)
    k = min(num_clusters, len(snippets))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    
    selected = []
    for cluster in range(k):
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster]
        centroid = kmeans.cluster_centers_[cluster]
        distances = [np.linalg.norm(embeddings[i] - centroid) for i in cluster_indices]
        best_index = cluster_indices[np.argmin(distances)]
        selected.append(snippets[best_index])

    return selected


# Funzione per elaborare un singolo file, numero di cluster fissato euristicamente a 3
def process_file(file_path, keywords, context, embedding_model, num_clusters=3):
    """
    Elabora un singolo file: legge il contenuto, estrae gli snippet e applica il clustering.
    Ritorna una tupla (file_path, snippet_rappresentativo) oppure None in caso di problemi.
    """
    if not os.path.exists(file_path) or not file_path.endswith(".py"):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Errore nella lettura del file {file_path}: {e}")
        return None
    
    print(f"Elaborazione file {file_path}")
    snippets = extract_relevant_ast_with_context(content, keywords, context)
    print(f"Numero di snippet estratti: {len(snippets)}")
    if not snippets:
        return None
    best_snippets_cluster = select_best_snippets_by_clustering(snippets, embedding_model, num_clusters=num_clusters)
    print(f"Numero di snippet rappresentativi selezionati: {len(best_snippets_cluster)}")
    return (file_path, best_snippets_cluster)


# Carica il JSON contenente i percorsi dei file filtrati
json_file_path = "files_median.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# Definisci le keyword di interesse
keywords = [
    "import", "from", "transformers", "huggingface", "AutoModel", "AutoTokenizer", "AutoConfig", "pipeline",
    "train", "fit", "fine_tune", "finetune",
    "token", "tokenizer", "encode", "decode",
    "predict", "inference", "generate", "metric",
    "sklearn", "scikit-learn"
]

# Carica il modello di embedding una sola volta
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dizionario per salvare i risultati
results = {}

# Numero di thread da utilizzare
max_workers = 4

# Processare i file in parallelo
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_info = {}
    
    # Itera su ogni cartella (o modello) e sui rispettivi file
    for model, file_paths in list(data.items())[:1]:
        print(f"Elaborazione modello {model}")
        if model not in results:
            results[model] = {}
        
        for file_path in file_paths[:2]:
            future = executor.submit(process_file, file_path, keywords, 2, embedding_model, num_clusters=3)
            future_to_info[future] = (model, file_path)
    
    # Raccogli i risultati
    for future in concurrent.futures.as_completed(future_to_info):
        model, file_path = future_to_info[future]
        try:
            res = future.result()
            if res is not None:
                # res è una tupla (file_path, snippet_rappresentativo)
                results[model][file_path] = res[1]
        except Exception as e:
            print(f"Errore elaborando il file {file_path}: {e}")


# Salva i risultati in un file JSON
output_file = "./best_snippets.json"

# Carica i risultati esistenti, se il file esiste
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as infile:
        existing_results = json.load(infile)
else:
    existing_results = {}

existing_results.update(results)

with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(existing_results, f_out, indent=4)

print(f"\nI migliori snippet sono stati salvati in '{output_file}'")
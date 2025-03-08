from huggingface_hub import InferenceClient, login
from transformers import AutoTokenizer
import random
import json
import time
import os
import re

# API client e autenticazione
api_key = "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"  # 1° profilo
#api_key = "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"

client = InferenceClient(provider="hf-inference", api_key=api_key)
login(api_key)

# JSON contenente gli snippet
with open("./best_snippets.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open('./models_without_output.json', 'r') as f:
    models_to_do = json.load(f)


# Impostazioni per il conteggio dei token e per il prompt
max_total_tokens = 4096
reserved_output_tokens = 400  # Risposta dell'LLM
temperatura = 0.2

# Modello LLM da utilizzare
llm_model = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model)

# Dizionario per raccogliere i risultati strutturati di ogni chiamata API
structured_results = {}

# Pattern per 'def' e 'import' come parole intere
pattern_def = re.compile(r"\bdef\b", flags=re.IGNORECASE)
pattern_import = re.compile(r"\bimport\b", flags=re.IGNORECASE)

# Itera sui modelli
# da fare da 200 a 300 su best 2
for snippet_model, files in list(data.items())[1000:]:


    # 'snippet_model'(es. "sentence-transformers_all-MiniLM-L6-v2")
    all_snippets = []

    # Sostituisce '_' con '/' per ottenere il nome corretto del modello
    snippet_model = snippet_model.replace("_", "/")
    print(f"Processing model: {snippet_model}")

    # Pattern per il nome del modello (assicurati di escapare se necessario)
    pattern_model = re.compile(r"\b" + re.escape(snippet_model) + r"\b", flags=re.IGNORECASE)
    
    # Prompt di sistema includendo il nome del modello di snippet
    system_prompt = (
        "You are an AI assistant specialized in Python. "
        "I will provide you with a series of code snippets extracted from various Python files. "
        "Note that these snippets might not form a coherent or complete code module when combined together. "
        "Your task is to analyze these snippets and extract recurring code patterns, such as function definitions (using 'def'), "
        "common imports, and other typical structures found in Python code. "
        f"Focus solely on identifying and returning the common patterns in the context of the model '{snippet_model}'.\n"
        "Return ONLY the code snippets. "
        "DO NOT include only the common imports."
        "Do NOT include any explanations, descriptions, or metadata. "
        "Do NOT generate any summaries, bullet points, markdown formatting, or additional text. "
        "If you are unable to find any relevant patterns, please return an empty string."
    )

    # Itera sui file per raccogliere gli snippet rilevanti
    for file_path, snippets in files.items():
        for snippet in snippets:
            # Filtra gli snippet: deve contenere il nome del modello o la parola "def"
            if not pattern_model.search(snippet) or not pattern_def.search(snippet):
                continue
            header = f"File: {file_path}\n"
            all_snippets.append(header + snippet)
            # file_paths_used.append(file_path)
    
    print(f"Lunghezza {len(all_snippets)}")

    # Seleziona un sottoinsieme di snippet (fino a 15) in maniera randomica se ce ne sono molti
    if len(all_snippets) >= 15:
        selected_snippets = random.sample(all_snippets, 15)
    else:
        selected_snippets = all_snippets

    # Calcola quanti token occupa il prompt di sistema
    system_tokens = len(tokenizer(system_prompt)["input_ids"])
    allowed_user_tokens = max_total_tokens - system_tokens - reserved_output_tokens

    # Costruisci il contenuto del messaggio utente in modo incrementale
    user_content = ""  
    for snippet in selected_snippets:
        candidate_text = user_content + ("\n" if user_content else "") + snippet
        candidate_tokens = len(tokenizer(candidate_text)["input_ids"])
        if candidate_tokens > allowed_user_tokens:
            break
        user_content = candidate_text

    #print(user_content)
    # Costruisci i messaggi per l'LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    # Verifica il conteggio dei token per il prompt complessivo
    system_tokens = len(tokenizer(messages[0]["content"])["input_ids"])
    user_tokens = len(tokenizer(messages[1]["content"])["input_ids"])
    total_input_tokens = system_tokens + user_tokens
    print(f"Total input tokens for snippet model '{snippet_model}': {total_input_tokens}")

    # Effettua la chiamata all'API LLM con logica di ripetizione:
    # Ripeti finché l'output non contiene il nome del modello e almeno una parola chiave ("def" o "import")
    attempts = 0
    max_attempts = 3  # tentativi
    while True:
        completion = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=reserved_output_tokens,
            temperature=0.2, 
            top_p=1.0
        )
        
        llm_output = completion.choices[0].message.content

        # se l'output contiene il nome del modello e almeno "def" o "import"
        if pattern_model.search(llm_output) or (pattern_def.search(llm_output) and pattern_import.search(llm_output)):
            print(f"Valid output generated")
            break  # output valido: esce dal loop

        attempts += 1
        if attempts >= max_attempts:
            print(f"Max attempts ({max_attempts}) reached for '{snippet_model}'. Using last output.")
            break  # esce dal loop anche se l'output non è perfetto

        print(f"Output non valido per '{snippet_model}'. Riprovo (tentativo {attempts}/{max_attempts})...")
        time.sleep(0.5)

    print(f"LLM output for snippet model '{snippet_model}':\n{llm_output}\n")

    # Registra in modo strutturato i dati relativi alla chiamata API
    structured_results[snippet_model] = {
        "system_prompt": system_prompt,
        "user_input": user_content,
        "llm_output": llm_output,
        "input_tokens": total_input_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "attempts": attempts,
        "temperature": temperatura
    }

    time.sleep(0.5)

# Percorso del file dei risultati
results_file = "LLM_results2.json"

# Carica i risultati esistenti, se il file esiste
if os.path.exists(results_file):
    with open(results_file, "r", encoding="utf-8") as infile:
        existing_results = json.load(infile)
else:
    existing_results = {}

# Aggiorna i risultati esistenti con i nuovi (le chiavi sono univoche, quindi non ci saranno conflitti)
existing_results.update(structured_results)

# Salva i risultati aggiornati
with open(results_file, "w", encoding="utf-8") as outfile:
    json.dump(existing_results, outfile, indent=4, ensure_ascii=False)
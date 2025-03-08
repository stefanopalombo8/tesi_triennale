import json
import requests
import time
from bs4 import BeautifulSoup

def get_python_code_from_huggingface(url):
    headers = {"User-Agent": "Mozilla/5.0"}  # per evitare blocchi
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Errore: impossibile accedere alla pagina ({response.status_code})")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # cerca i tag <pre><code> che contengono codice
    code_blocks = []
    for code_tag in soup.find_all("code"):  # Trova tutti i tag <code>
        code_text = code_tag.get_text()
        
        # codice Python
        if "import " in code_text or "def " in code_text or "from " in code_text:
            code_blocks.append(code_text)
    
    return code_blocks

if __name__ == "__main__":

    with open('./model-num_files.json', mode="r") as f:
        data = json.load(f)

    models_name = list(model for model, num in data.items() if num >= 100)

    print(len(models_name))

    result = {}

    for model in models_name:
        result[model] = []
        url = "https://huggingface.co/" + model
        python_code_snippets = get_python_code_from_huggingface(url)

        if python_code_snippets:
            print("Codice Python trovato:")
            for snippet in python_code_snippets:
                result[model].append(snippet)
        else:
            print("Nessun codice Python trovato.")
        
        time.sleep(0.05)
    
    with open('./HF_model_cards.json', mode="w") as f:
        json.dump(result, f, indent=4)
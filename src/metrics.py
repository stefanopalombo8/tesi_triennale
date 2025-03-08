import json
import os
from huggingface_hub import InferenceClient, login
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.meteor_score import meteor_score
from codebleu import calc_codebleu

#nltk.download('wordnet')

api_key = "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"  # 1° profilo
#login(api_key)

def trasforma_nome(nome):
    if '/' in nome:
        # Divido in due parti: prima parte e resto
        parte_iniziale, resto = nome.split('/', 1)
        # Sostituisco gli slash nel resto con underscore
        return parte_iniziale + '/' + resto.replace('_', '/')
    return nome

def compute_meteor(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return meteor_score([ref_tokens], cand_tokens)

def compute_code_bleu(reference, candidate):
    return calc_codebleu([reference], [candidate], "python")

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    # Esegui il mean pooling sui token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()

def compute_cosine_similarity(reference, candidate, tokenizer, model):
    embedding_ref = get_embedding(reference, tokenizer, model)
    embedding_cand = get_embedding(candidate, tokenizer, model)
    similarity = cosine_similarity(embedding_ref, embedding_cand)[0][0]
    return similarity

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # model = AutoModel.from_pretrained("microsoft/codebert-base")

    with open("models_name.json", "r") as f:
        models = json.load(f)

    models_name = [model_name.replace("_", "/", 1) for model_name in models]
    #models_name = [trasforma_nome(model_name) for model_name in models_name]

    models_name = models_name[:1064]

    with open("LLM_results.json", "r", encoding="utf-8") as f:
        LLM_results = json.load(f)

    with open("HF_model_cards.json", "r", encoding="utf-8") as f:
        HF_results = json.load(f)

    metrics = {}
    
    # facebook/esm2_t12_35M_UR50D HF_cards
    # facebook/esm2/t12/35M/UR50D LLM_results

    for m in HF_results.keys():
        try:
            HF_cards = HF_results[m] # questa lista ha i nomi corretti 
            
            # if HF_cards == []:
            #     continue

            m = trasforma_nome(m)

            # HF_card = "\n\n".join(HF_cards)
            LLM_card = LLM_results[m] # questa lista non ha i nomi corretti

            # if LLM_card.strip() == "":
            #     continue
            
            # meteor = compute_meteor(HF_card, LLM_card)
            # #print("Meteor:", meteor)
            
            #code_bleu = compute_code_bleu(HF_card, LLM_card)
            # print("CodeBlue:", code_bleu)

            # cosine_sim = compute_cosine_similarity(HF_card, LLM_card, tokenizer, model)
            # #print("Cosine similarity:", cosine_sim)
            
            # code_bleu_metrics = []
            # for metric in code_bleu:
            #     score = float(code_bleu[metric])
            #     code_bleu_metrics.append({metric:score})
            
            # metrics[m] = {
            #     "code_bleu": code_bleu_metrics,
            #     "cosine_similarity": float(cosine_sim),
            #     "meteor": float(meteor)
            # }

            
        except Exception as e:
            print(f"Errore {e} per il modello {m}")
            continue

    # Salva i risultati in un file JSON
    #with open("./metrics.json", "w", encoding="utf-8") as json_file:
    #    json.dump(metrics, json_file, indent=4)
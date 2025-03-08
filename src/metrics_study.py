import json

def trasforma_nome(nome):
    if '/' in nome:
        # Divido in due parti: prima parte e resto
        parte_iniziale, resto = nome.split('/', 1)
        # Sostituisco gli slash nel resto con underscore
        return parte_iniziale + '/' + resto.replace('/', '_')
    return nome

with open("HF_model_cards.json", "r", encoding="utf-8") as f:
    HF_results = json.load(f)

models_with_cards = [model for model in HF_results.keys() if HF_results[model] != []]
print("modelli che su HF hanno codice ", len(models_with_cards))

with open("LLM_results_final.json", "r", encoding="utf-8") as f:
    LLM_results = json.load(f)

models_with_output = [trasforma_nome(model) for model in LLM_results.keys() if LLM_results[model]["llm_output"].strip() != ""]
print("modelli che l'LLM ha prodotto codice", len(models_with_output))

models_to_compare = list(set(models_with_output).intersection(set(models_with_cards)))
print("modelli di cui posso fare metriche ", len(models_to_compare))

# modelli che hanno output ma non hanno cards
models_without_cards = list(set(models_with_output).difference(set(models_with_cards)))
#print(models_without_cards[:3])
print("modelli che hanno output ma non hanno cards ", len(models_without_cards))

models_without_cards = [model for model in models_with_output if model not in models_with_cards]
print("modelli che hanno output ma non hanno cards ", len(models_without_cards))

#print(models_without_cards)


# CALCOLO RISULTATI METRICHE 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("./metrics2.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

print(len(metrics))

results = []
for model, values in metrics.items():
    # Estrai il valore principale di code_bleu (che è il primo elemento)

    # Estrazione dei punteggi
    ngram_match_score = values["code_bleu"][1]["ngram_match_score"]
    weighted_ngram_match_score = values["code_bleu"][2]["weighted_ngram_match_score"]
    syntax_match_score = values["code_bleu"][3]["syntax_match_score"]
    dataflow_match_score = values["code_bleu"][4]["dataflow_match_score"]

    weights = {
        "ngram_match_score": 0.05,
        "weighted_ngram_match_score": 0.05,
        "syntax_match_score": 0.45,
        "dataflow_match_score": 0.45
    }


    weighted_codebleu = (
        weights["ngram_match_score"] * ngram_match_score +
        weights["weighted_ngram_match_score"] * weighted_ngram_match_score +
        weights["syntax_match_score"] * syntax_match_score +
        weights["dataflow_match_score"] * dataflow_match_score
    )

    #print(weighted_codebleu)

    round(weighted_codebleu, 2)

    #weighted_codebleu = weighted_codebleu / 4

    # Estrai le altre metriche
    cosine_similarity = values["cosine_similarity"]
    meteor = values["meteor"]
    
    results.append({
        "model": model,
        "code_bleu": weighted_codebleu,
        "cosine_similarity": cosine_similarity,
        "meteor": meteor
    })


import scipy.stats as stats
import numpy as np

# Estraiamo i valori delle metriche in tre liste
code_bleu_vals = [item["code_bleu"] for item in results]
cosine_vals = [item["cosine_similarity"] for item in results]
meteor_vals = [item["meteor"] for item in results]

code_bleu_vals_mean = np.mean(code_bleu_vals)
print("MEDIA CODEBLUE ", code_bleu_vals_mean)
cosine_vals_mean = np.mean(cosine_vals)
print("MEDIA COSINE ", cosine_vals_mean)
meteor_vals_vals_mean = np.mean(meteor_vals)
print("MEDIA METEOR ", meteor_vals_vals_mean)


# # Creiamo tre istogrammi affiancati
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Funzione per plottare l'istogramma e la distribuzione

# def plot_histogram_with_kde(ax, data, color, title, xlabel):
#     ax.hist(data, bins=20, color=color, alpha=0.7, density=True, edgecolor='black')
#     kde = stats.gaussian_kde(data)
#     x_vals = np.linspace(min(data), max(data), 100)
#     ax.plot(x_vals, kde(x_vals), color='red', linestyle='dashed', linewidth=2)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Densità")

# # Istogramma Code BLEU
# plot_histogram_with_kde(axes[0], code_bleu_vals, "blue", "Distribuzione Code BLEU", "Valori Code BLEU")

# # Istogramma Cosine Similarity
# plot_histogram_with_kde(axes[1], cosine_vals, "green", "Distribuzione Cosine Similarity", "Valori Cosine Similarity")

# # Istogramma Meteor
# plot_histogram_with_kde(axes[2], meteor_vals, "orange", "Distribuzione Meteor", "Valori Meteor")

# plt.tight_layout()
# plt.show()

# # ###############################################
# # # 3. Violin Plot
# # ###############################################
# fig, ax = plt.subplots(figsize=(8, 6))
# data_to_plot = [code_bleu_vals, cosine_vals, meteor_vals]
# parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)

# # Colorazione
# colors = ["blue", "green", "orange"]
# for i, body in enumerate(parts['bodies']):
#     body.set_facecolor(colors[i])
#     body.set_edgecolor("black")
#     body.set_alpha(0.7)

# ax.set_title("Violin Plot delle Metriche")
# ax.set_xticks([1, 2, 3])
# ax.set_xticklabels(["Code BLEU", "Cosine Similarity", "Meteor"])
# ax.set_ylabel("Valori")

# plt.show()

# ###############################################
# # 4. Pair Plot (Scatter Matrix)
# ###############################################
# # Creiamo un DataFrame dai risultati
# df = pd.DataFrame(results)
# # Selezioniamo solo le colonne numeriche
# df_num = df[["code_bleu", "cosine_similarity", "meteor"]]

# import matplotlib.ticker as mticker
# # Creiamo la scatter matrix (pair plot)
# axes = pd.plotting.scatter_matrix(df_num, figsize=(8, 8), diagonal='kde')
# colors = ["blue", "green", "orange"]  # Ordine: [code_bleu, cosine_similarity, meteor]

# for i in range(len(df_num.columns)):
#     for j in range(len(df_num.columns)):
#         ax = axes[i, j]
        
#         # Se siamo sulla diagonale (KDE)
#         if i == j:
#             # Se c'è la linea della KDE
#             if ax.lines:
#                 ax.lines[0].set_color(colors[i])
        
#         # Se siamo in uno scatter
#         else:
#             # ax.collections di solito contiene i 'PathCollection' dei punti
#             for col in ax.collections:
#                 # Ad esempio, coloriamo in base alla riga 'i'
#                 # (cioè la variabile sul "perno" delle Y)
#                 col.set_color(colors[i])
#                 # Se preferisci colorare in base a 'j' (variabile su X),
#                 # puoi fare col.set_color(colors[j])

# plt.show()
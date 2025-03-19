import json

with open("LLM_results_final.json", "r", encoding="utf-8") as f:
    LLM_results = json.load(f)

print(LLM_results["blanchefort/rubert-base-cased-sentiment"]["llm_output"])

# with open("HF_model_cards.json", "r", encoding="utf-8") as f:
#     HF_results = json.load(f)

# for model in HF_results.keys():
#     if HF_results[model] == []:
#         print(model)

# # models_without_output = [model for model in LLM_results.keys() if LLM_results[model]["llm_output"].strip() == ""]

# for model in LLM_results.keys():
#     if "AutoModel" in LLM_results[model]["llm_output"] :
#         print(LLM_results[model]["llm_output"])
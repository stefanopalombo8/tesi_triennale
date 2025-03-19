import json

with open("./file_json/model-num_files_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(sum(data.values()))
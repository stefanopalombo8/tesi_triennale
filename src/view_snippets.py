import json

with open("./best_snippets.json", "r", encoding="utf-8") as f:
    data = json.load(f)

number_of_snippet = {}


for snippet_model, files in data.items():
    for file, snippets in files.items():
        if len(snippets) not in number_of_snippet:
            number_of_snippet[len(snippets)] = 0
        number_of_snippet[len(snippets)] += 1

print(number_of_snippet)
# {2: 29867, 3: 69438, 1: 26375}
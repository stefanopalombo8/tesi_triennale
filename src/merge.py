import json

def merge_json(json1, json2):
    result = {}
    for key in json1.keys():
        if key in json2:
            result[key] = json1[key] + json2[key]
        else:
            result[key] = json1[key]
    for key in json2.keys():
        if key not in json1:
            result[key] = json2[key]
    return result

def main():
    with open('./best_snippets.json', 'r', encoding="utf-8") as f:
        json1 = json.load(f)
    with open('./best_snippets2.json', 'r', encoding="utf-8") as f:
        json2 = json.load(f)
    
    print(len(list(json1.keys())))
    print(len(list(json2.keys())))

    result = merge_json(json1, json2)
    with open('./best_snippets_final.json', 'w') as f:
        json.dump(result, f)

#main()

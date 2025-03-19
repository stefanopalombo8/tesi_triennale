import os

def get_files_per_folder(root_dir):
    """
    Crea un dizionario con la struttura:
    {nome_cartella: set(nomi_file)}
    """
    folder_files = {}
    for foldername in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            folder_files[foldername] = len(os.listdir(folder_path))
    return folder_files

def find_extra_files(root_dir1, root_dir2):
    """
    Trova la cartella in cui esiste un file in più tra root_dir1 e root_dir2.
    """
    files1 = get_files_per_folder(root_dir1)
    files2 = get_files_per_folder(root_dir2)

    all_folders = set(files1.keys()).union(set(files2.keys()))

    for folder in all_folders:
        set1 = files1.get(folder, set())
        set2 = files2.get(folder, set())

        extra_in_1 = set1 - set2  # File presenti solo in root_dir1
        extra_in_2 = set2 - set1  # File presenti solo in root_dir2

        if extra_in_1:
            print(f"📂 Differenza in '{folder}' (solo in {root_dir1}): {extra_in_1}")
        if extra_in_2:
            print(f"📂 Differenza in '{folder}' (solo in {root_dir2}): {extra_in_2}")

if __name__ == "__main__":
    root_directory_1 = r"C:\Users\stefa\Desktop\models_files_drive\models_files"  
    root_directory_2 = r"C:\Users\stefa\Desktop\models_files"  

    import json
    with open("./file_json/model-num_files_final.json", "w") as f:
        json.dump(get_files_per_folder(root_directory_1), f, indent=4) 

    #find_extra_files(root_directory_1, root_directory_2)
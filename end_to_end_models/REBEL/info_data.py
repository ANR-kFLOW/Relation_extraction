import os

def count_redundancies(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        unique_lines = set(lines)
        num_redundancies = len(lines) - len(unique_lines)
        return len(lines), num_redundancies

def main():
    folder_path = "/data/Youss/RE/REBEL/data/news_data_with_cnc"
    files = os.listdir(folder_path)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            length, redundancies = count_redundancies(file_path)
            print(f"File: {file_name}, Length: {length}, Redundancies: {redundancies}")

if __name__ == "__main__":
    main()


import os
import json
import csv

# Percorso della cartella principale
root_dir = "training/forecasting/kernel_size_96_valid"

# Cerca tutti i file valid_best_score.json nella directory e sottodirectory
for subdir, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name == "valid_best_score.json":
            # Percorso completo del file JSON
            json_file = os.path.join(subdir, file_name)

            # Percorso per salvare il file CSV (nella stessa directory del JSON)
            csv_file = os.path.join(subdir, "valid_best_score.csv")

            # Leggi il file JSON
            with open(json_file, 'r') as file:
                data = json.load(file)

            # Scrivi i dati nel file CSV
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Prediction Length", "Value"])  # Intestazioni
                for key, value in data.items():
                    writer.writerow([key, value])

            print(f"File CSV salvato come {csv_file}")

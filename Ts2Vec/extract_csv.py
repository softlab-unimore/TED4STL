import os
import re
import json
import pandas as pd
from argparse import ArgumentParser

def find_eval_res_files(start_path):
    """
    Find all the eval_res.json files in the specified directory.
    """
    matches = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file == 'eval_res.json':
                matches.append(os.path.join(root, file))
    return matches

def extract_data_from_file(file_path, data_type):
    """
    Extracts the accuracy and loss from the eval_res.json file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            if 'forecasting' in args.directory:
                data = data['ours']
                match = re.search(r'([^/]+)__forecast_multivar_', file_path).group(1)
            else:
                raise ValueError(f"Unknown task type")

            splits = file_path.split('/')
            model_name = splits[4]
            extract_data = {model_name:{match: {}}}
            if 'forecasting' in args.directory:
                for key in data.keys():
                    extract_data[model_name][match][key] ={
                        'MAE': round(data[key][data_type]['MAE'], 4),
                        'MSE': round(data[key][data_type]['MSE'], 4)
                    }
            else:
                raise ValueError(f"Unknown task type")
            return extract_data
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None

def extract_rows_forecasting(data_list):
    rows = []
    for data in data_list:
        for model_name, predictions in data.items():
            for dataset_name, len_pred in predictions.items():
                for pred_length, metrics in len_pred.items():
                    rows.append({
                        'Model': model_name,
                        'Dataset': dataset_name,
                        'Prediction Length': int(pred_length),
                        f'MAE': metrics['MAE'],
                        f'MSE': metrics['MSE']
                    })

    df = pd.DataFrame(rows).sort_values(by=['Model', 'Dataset', 'Prediction Length'])
    df = df.pivot_table(index=['Dataset', 'Prediction Length'], columns='Model', values=['MAE', 'MSE']).round(4)
    df.columns = [f'{model}_{metric}' for metric, model in df.columns]
    df = df.reindex(sorted(df.columns), axis=1)
    df.reset_index(inplace=True)
    df['Dataset'] = df['Dataset'].where(df['Dataset'] != df['Dataset'].shift(), " ")
    return df

def main(directory, output_csv, type):
    """
    Extracts MAE e MSE from eval_res.json files and saves them in a CSV file.
    """
    files = find_eval_res_files(directory)
    data_list = [extract_data_from_file(file, type) for file in files]
    data_list = [data for data in data_list if data is not None]  # Rimuovi eventuali None

    if 'forecasting' in args.directory:
        df = extract_rows_forecasting(data_list)
    else:
        raise ValueError(f"Unknown task type")

    df.to_csv(f'{directory}/{output_csv}', index=False)
    print(f"Data saved in {args.directory}/{output_csv}")

if __name__ == "__main__":
    parser = ArgumentParser(description='Insertion of correct path to save the csv')
    parser.add_argument('--directory', type=str, help='Directory where the eval_res.json files are located')
    parser.add_argument('--type', type=str, default='norm', help='Choose if you want raw or norm data')
    args = parser.parse_args()
    output_csv = "results.csv" if args.type == 'norm' else "results_raw.csv"
    if args.type == 'norm':
        output_csv = "results.csv"  # Cambia il nome del file CSV se necessario
    else:
        output_csv = "results_raw.csv"
    main(f'./training/{args.directory}', output_csv, args.type)

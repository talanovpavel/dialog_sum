import csv
from pathlib import Path
import yaml

from vllm import LLM
from vllm import SamplingParams

def log_results(results_path,
                name:str,
                row:dict):
    """
    Write one row of experiment to CSV file
    """
    row['name'] = name

    results_path = Path(results_path)

    file_exists = results_path.is_file()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def load_config(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
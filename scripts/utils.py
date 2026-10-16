import csv
import pathlib

def log_results(results_path,
                row:dict):
    """
    Write one row of experiment to CSV file
    """

    file_exists = results_path.is_file()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


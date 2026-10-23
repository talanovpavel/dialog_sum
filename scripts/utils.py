import csv
from pathlib import Path
import yaml
import pandas as pd
import time
from vllm import LLM
from vllm import SamplingParams
import logging

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


def make_llm_results_dict(
        conversations,
        outputs
):
    rows = []
    texts = [t[-1]['content'] for t in conversations]

    for text, out in zip(texts, outputs):
        rows.append({
        "text": text,
        "output": out.outputs[0].text.strip()
        })

    return rows

def make_results_table(
        rows,
        test_path
):

    test = pd.read_csv(test_path)
    result_df = pd.DataFrame(rows)

    return pd.merge(test, result_df, how='left', left_on='dialogue', right_on='text').drop(columns='text')

def get_outputs(conversations, llm_params, sampling_params, logger):
    t = time.perf_counter()
    logger.info("Loading model: %s", llm_params.get('model'))
    llm = LLM(**llm_params)
    logger.info("Model loaded (%.1f s)", time.perf_counter() - t)

    t = time.perf_counter()
    logger.info("Generating outputs for %d dialogues...", len(conversations))
    outputs = llm.chat(
        conversations,
        SamplingParams(**sampling_params),
        use_tqdm=True,
    )
    logger.info("Generation finished (%.1f s)", time.perf_counter() - t)

    return outputs
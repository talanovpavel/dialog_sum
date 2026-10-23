import logging

from vllm import LLM
from vllm import SamplingParams

from scripts.metrics import count_metrics
from scripts.utils import log_results, load_config

import click
import time

import pandas as pd

logger = logging.getLogger(__name__)

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

def get_outputs(conversations, llm_params, sampling_params):
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


def make_prompts(train_path,
                 test_path,
                 system='Summarize this text. Use given examples',
                 few_shot=True,
                 n_examples=5,
                 dialogue_col='dialogue',
                 summary_col='summary'
                 ):
    test = pd.read_csv(test_path)
    train = pd.read_csv(train_path)

    train_unique = train.drop_duplicates([dialogue_col])
    test_unique = test.drop_duplicates([dialogue_col])

    if few_shot:

        FEW_SHOT = [
            msg
            for n in range(n_examples)
            for msg in (
                {"role": "user", "content": train_unique.iloc[n][dialogue_col]},
                {"role": "assistant", "content": train_unique.iloc[n][summary_col]},
            )
        ]

        conversations = [
            [{"role": "system", "content": system},
             *FEW_SHOT,
             {"role": "user", "content": t}]
            for t in test_unique.dialogue
        ]

        return conversations

    else:

        conversations = [
            [{"role": "system", "content": system},
             {"role": "user", "content": t}]
            for t in test_unique.dialogue
        ]

        return conversations


@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    t_start = time.perf_counter()

    logger.info("[1/6] Loading config: %s", config_path)
    config = load_config(config_path)

    logger.info("[2/6] Building prompts")
    conversations = make_prompts(train_path=config['data']['train_path'],
                                 test_path=config['data']['test_path'],
                                 **config['prompt'])
    logger.info("Prompts created: %d", len(conversations))

    logger.info("[3/6] Running inference")
    outputs = get_outputs(conversations, config['llm'], config['sampling'])

    logger.info("[4/6] Collecting LLM results")
    rows = make_llm_results_dict(conversations, outputs)

    logger.info("[5/6] Building results table and computing metrics")
    res_table = make_results_table(rows, config['data']['test_path'])
    logger.info("Results table: %d rows", len(res_table))
    res_table.to_csv(config['data']['output_path'], index=False)
    logger.info("Outputs table saved to %s", config['data']['output_path'])
    metrics = count_metrics(res_table, hyps_col='output', refs_col='summary')
    logger.info("LLM metrics: %s", metrics)

    logger.info("[6/6] Saving results")
    log_results(config['result_path'], config['name'], metrics)
    logger.info("Results saved to %s", config['result_path'])
    logger.info("Done in %.1f s", time.perf_counter() - t_start)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
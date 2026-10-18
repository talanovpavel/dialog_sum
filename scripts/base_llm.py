import logging

from vllm import LLM
from vllm import SamplingParams
import pandas as pd

from scripts.metrics import count_metrics
from scripts.utils import log_results, load_config

import click

logger = logging.getLogger(__name__)

def make_prompts(train_path,
                 test_path,
                 system='Summarize this text. Use given examples',
                 n_examples=5,
                 dialogue_col='dialogue',
                 summary_col='summary'
                 ):

    test = pd.read_csv(test_path)
    train = pd.read_csv(train_path)

    train_unique = train.drop_duplicates([dialogue_col])
    test_unique = test.drop_duplicates([dialogue_col])

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

def get_outputs(conversations, llm_params, sampling_params):
    llm = LLM(
        **llm_params
    )

    outputs = llm.chat(
        conversations,
        SamplingParams(**sampling_params)
    )

    return outputs

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


@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    config = load_config(config_path)

    conversations = make_prompts(train_path=config['data']['train_path'],
                                 test_path=config['data']['test_path'],
                                 **config['prompt'])

    outputs = get_outputs(conversations, config['llm'], config['sampling'])

    rows = make_llm_results_dict(conversations, outputs)

    res_table = make_results_table(rows, config['data']['test_path'])

    metrics = count_metrics(res_table, hyps_col='output', refs_col='summary')
    logger.info("LLM metrics: %s", metrics)

    log_results(config['result_path'], metrics)
    logger.info("Results saved to %s",
                config['result_path'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
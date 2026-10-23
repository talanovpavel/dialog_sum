import click
from nltk.tokenize import sent_tokenize
import pandas as pd
import logging

from scripts.metrics import count_metrics
from scripts.utils import log_results, load_config

logger = logging.getLogger(__name__)

def baseline(dialogue):
    """Create a summary from the first and last sentences of a dialogue."""

    sents = sent_tokenize(dialogue.strip())

    if not sents:
        return ""

    return sents[0] if len(sents) == 1 else sents[0] + ' ' + sents[-1]

def apply_baseline(df,
                   dialogue_col:str
                   ):
    """Apply the baseline summarization to a DataFrame column."""

    df = df.copy()
    df['baseline'] = df[dialogue_col].apply(baseline)

    return df

@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    config = load_config(config_path)

    df = pd.read_csv(config['test_path'])
    df_baseline = apply_baseline(df,
                                 dialogue_col=config['dialogue_col'])
    df_baseline.to_csv(config['baseline_path'])
    logger.info("Baseline saved to %s",
                config['baseline_path'])

    result = count_metrics(df_baseline, hyps_col=config['hyps_col'], refs_col=config['refs_col'])
    logger.info("Baseline metrics: %s", result)
    result['name'] = 'baseline'

    log_results(config['result_path'], config['name'], result)
    logger.info("Results saved to %s",
                config['result_path'])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
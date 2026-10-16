from nltk.tokenize import sent_tokenize
import pandas as pd
import logging

from scripts.config import TEST_CLEAN_PATH, BASELINE_PATH, RESULTS_PATH
from scripts.metrics import count_metrics
from scripts.utils import log_results

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

def run():
    df = pd.read_csv(TEST_CLEAN_PATH)
    df_baseline = apply_baseline(df,
                                 dialogue_col="dialogue")
    df_baseline.to_csv(BASELINE_PATH)
    logger.info("Baseline saved to %s",
                BASELINE_PATH)

    result = count_metrics(df_baseline, hyps_col="baseline", refs_col="summary", results_path=RESULTS_PATH)
    logger.info("Baseline metrics: %s", result)
    result['name'] = 'baseline'

    log_results(RESULTS_PATH, result)
    logger.info("Results saved to %s",
                RESULTS_PATH)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
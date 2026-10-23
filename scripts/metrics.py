import sacrebleu
from rouge_score import rouge_scorer
import numpy as np

from scripts.utils import log_results

def bleu(
        df,
        hyps_col:str,
        refs_col:str,
)-> float:
    """
    Corpus BLEU (sacrebleu) per dialog:
    dialog = id without last '_N';
    hypothesis = first in dialog;
    references = all rows of the dialog (padded by repeating the last one to equal count)
    """
    df['dialog_id'] = df['id'].str.rsplit('_', n=1).str[0]

    g = df.groupby('dialog_id', sort=False)
    hyps = g[hyps_col].first().astype(str).tolist()
    refs_lists = g[refs_col].apply(list).tolist()

    n_refs = max(len(r) for r in refs_lists)
    padded = [r + [r[-1]] * (n_refs - len(r)) for r in refs_lists]
    refs = [[r[i] for r in padded] for i in range(n_refs)]

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score

def rouge(
        df,
        hyps_col: str,
        refs_col: str
)-> dict:
    """
    Mean ROUGE-1/2/L (stemmed F-measure) over samples; per sample, takes the max over its references;
    returns a dict {metric: mean score}
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    res = {k: [] for k in ["rouge1", "rouge2", "rougeL"]}

    g = df.groupby('dialog_id', sort=False)
    hyps = g[hyps_col].first().astype(str).tolist()
    refs_lists = g[refs_col].apply(list).tolist()

    for hyp, rs in zip(hyps, refs_lists):
        scores = [scorer.score(r, hyp) for r in rs]
        for k in res:
            res[k].append(max(s[k].fmeasure for s in scores))

    result = {}

    for k, v in res.items():
        result[k] = np.mean(v)

    return result


def count_metrics(df, hyps_col, refs_col):

    bleu_score = bleu(df, hyps_col, refs_col)
    rouge_score = rouge(df, hyps_col, refs_col)

    result = rouge_score
    result['bleu'] = bleu_score

    return result
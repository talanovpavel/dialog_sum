import pandas as pd
import logging

from scripts.config import TEST_PATH, TEST_CLEAN_PATH

logger = logging.getLogger(__name__)

def remove_duplicates(df: pd.DataFrame,
                      cols: list[str]
                      )-> pd.DataFrame:
    """
    Remove duplicates from DataFrame based on cols
    """

    df = df.copy()
    clean = df.drop_duplicates(subset=cols).reset_index(drop=True)
    logger.info("Removed %d duplicates (%d -> %d rows)",
                len(df) - len(clean), len(df), len(clean))

    return clean

def run():
    df = pd.read_csv(TEST_PATH)
    clean = remove_duplicates(df, ['dialogue', 'summary'])
    clean.to_csv(TEST_CLEAN_PATH)
    logger.info("Preprocessed dataset saved to %s",
                TEST_CLEAN_PATH)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()



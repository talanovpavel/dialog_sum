import pandas as pd
import logging
import click

from scripts.utils import load_config

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame,
                      cols: list[str]
                      ) -> pd.DataFrame:
    """
    Remove duplicates from DataFrame based on cols
    """

    df = df.copy()
    clean = df.drop_duplicates(subset=cols).reset_index(drop=True)
    logger.info("Removed %d duplicates (%d -> %d rows)",
                len(df) - len(clean), len(df), len(clean))

    return clean


@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    config = load_config(config_path)

    df = pd.read_csv(config['test_path'])
    clean = remove_duplicates(df, config['cols'])
    clean.to_csv(config['test_clean_path'])
    logger.info("Preprocessed dataset saved to %s",
                config['test_clean_path'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


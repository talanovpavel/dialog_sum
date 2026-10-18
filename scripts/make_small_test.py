import pandas as pd
import click
import logging

logger = logging.getLogger(__name__)

def make_small_test(test_path):
    test = pd.read_csv(test_path)
    return test.head(15)

@click.command()
@click.option('--test_path', 'test_path', required=True)
@click.option('--small_test_path', 'small_test_path', required=True)
def main(test_path, small_test_path):
    small_test = make_small_test(test_path)

    small_test.to_csv(small_test_path, index=False)
    logger.info("Small version of test saved to %s",
                small_test_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

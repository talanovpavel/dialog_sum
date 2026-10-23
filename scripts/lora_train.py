import logging
import time

import click
import pandas as pd
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import Dataset

from scripts.utils import load_config

logger = logging.getLogger(__name__)


def load_dataset(train_path):
    return pd.read_csv(train_path)


def make_messages(train_df, INSTRUCTION):
    def to_messages(row):
        return [
            {"role": "user", "content": INSTRUCTION + row['dialogue']},
            {"role": "assistant", "content": row['summary']},
        ]

    train_df['messages'] = train_df.apply(to_messages, axis=1)
    return train_df


@click.command()
@click.option('--config', 'config_path', required=True)
def train(config_path):
    t_start = time.perf_counter()

    logger.info("[1/5] Loading config: %s", config_path)
    config = load_config(config_path)

    logger.info("[2/5] Loading dataset and building messages")
    train_df = load_dataset(config['data']['train_path'])
    logger.info("Train rows loaded: %d", len(train_df))
    train_df = make_messages(train_df, config['INSTRUCTION'])
    train_ds = Dataset.from_pandas(train_df[['messages']], preserve_index=False)
    logger.info("Dataset created: %d examples", len(train_ds))

    logger.info("[3/5] Loading model: %s", config['model'])
    model, tokenizer = FastLanguageModel.from_pretrained(config['model'])
    model = FastLanguageModel.get_peft_model(model)
    logger.info("LoRA adapters attached")

    logger.info("[4/5] Training")
    t_train = time.perf_counter()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
    )
    train_result = trainer.train()
    logger.info("Training finished in %.1f s", time.perf_counter() - t_train)
    logger.info("Train metrics: %s", train_result.metrics)

    logger.info("[5/5] Saving merged model to %s", config['save_path'])
    model.save_pretrained_merged(config['save_path'], tokenizer, save_method="merged_16bit")
    logger.info("Model saved to %s", config['save_path'])

    logger.info("Done in %.1f s", time.perf_counter() - t_start)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train()


    """
    туду:
    2) добавить логи
    3) добавить прогресс бар или подобную хуйнюшку(wandb)
    """
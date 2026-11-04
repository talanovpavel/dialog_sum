import json
import logging
import time
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from scripts.utils import load_config

logger = logging.getLogger(__name__)


def make_faithfulness_coverage_df(test_path, base_llm_path, finetuned_path,
                                  result_path, text_col='dialogue'):
    test = pd.read_csv(test_path, keep_default_na=False)
    test = test.drop_duplicates(text_col).reset_index(drop=True)
    test = test.loc[:, ~test.columns.str.startswith('Unnamed:')]

    if test.empty or test[text_col].str.strip().eq('').any():
        raise ValueError("Test data must contain non-empty dialogues")

    for name, path in [('base_llm', base_llm_path),
                       ('finetuned_llm', finetuned_path)]:
        outputs = pd.read_csv(path, keep_default_na=False)
        outputs = outputs[[text_col, 'output']].drop_duplicates()
        outputs = outputs.rename(columns={'output': name})
        test = test.merge(outputs, how='left', on=text_col, validate='one_to_one')

        if test[name].isna().any():
            raise ValueError(f"Missing summaries for {name}")

    test.to_csv(result_path, index=False)
    return test


def make_messages(system, user, data_path, text_col):
    data = pd.read_csv(data_path, keep_default_na=False)
    texts = data[text_col].unique().tolist()

    messages = [
        [{"role": "system", "content": system},
         {"role": "user", "content": user.replace("{text}", t)}]
        for t in texts if t.strip()
    ]

    return messages


def get_questions(model, messages, temperature, client=None,
                  request_delay=0):
    client = client if client is not None else Groq()
    questions = []

    for n, m in enumerate(messages, start=1):
        logger.info("Generating questions %d/%d", n, len(messages))
        time.sleep(request_delay)
        chat_completion = client.chat.completions.create(
            messages=m,
            model=model,
            temperature=temperature,
            response_format={'type': 'json_object'}
        )
        result = json.loads(chat_completion.choices[0].message.content)
        qs = result['questions']

        if not isinstance(qs, list) or any(
                not isinstance(q, str) or not q.strip() for q in qs
        ):
            raise ValueError("Judge must return a list of question strings")

        questions.append(list(dict.fromkeys(q.strip() for q in qs)))

    return questions


def get_answers(model, answer_prompt, text, questions, temperature,
                client=None, request_delay=0):
    """Get answers to all questions about one text in a single request."""
    if not questions:
        return []
    if not text.strip():
        return ['unknown'] * len(questions)

    client = client if client is not None else Groq()
    prompt = answer_prompt.format(
        text=text, questions=json.dumps(questions, ensure_ascii=False)
    )
    time.sleep(request_delay)
    chat_completion = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model=model,
        temperature=temperature,
        response_format={'type': 'json_object'}
    )
    result = json.loads(chat_completion.choices[0].message.content)
    answers = result['answers']

    if (not isinstance(answers, list) or len(answers) != len(questions)
            or any(not isinstance(a, str) for a in answers)):
        raise ValueError("Judge must return one answer per question, in the same order")

    answers = [a.strip().lower() for a in answers]
    if any(a not in ('yes', 'no', 'unknown') for a in answers):
        raise ValueError(f"Unexpected judge answers: {answers}")

    return answers


def count_score(source_answers, target_answers):
    matches = [
        source == target
        for source, target in zip(source_answers, target_answers, strict=True)
        if source != 'unknown'
    ]
    return sum(matches) / len(matches) if matches else float('nan')


def evaluate(df, config, client):
    rows = []
    params = {**config['model'], 'client': client}

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        logger.info("Evaluating dialogue %d/%d", n, len(df))
        result = row.to_dict()
        dialogue = row[config['data']['text_col']]
        questions = row['coverage_questions']
        answer_prompt = config['coverage']['answer_prompt']
        source_answers = get_answers(
            text=dialogue, questions=questions, answer_prompt=answer_prompt, **params
        )
        result['coverage_source_answers'] = source_answers

        for col in ['base_llm', 'finetuned_llm']:
            summary = row[col]
            summary_answers = get_answers(
                text=summary, questions=questions, answer_prompt=answer_prompt, **params
            )
            result[f'{col}_coverage'] = count_score(source_answers, summary_answers)
            result[f'{col}_coverage_answers'] = summary_answers

            claims = row[f'{col}_faithfulness_questions']
            answer_prompt_faithfulness = config['faithfulness']['answer_prompt']
            claim_answers = get_answers(
                text=summary, questions=claims,
                answer_prompt=answer_prompt_faithfulness, **params
            )
            dialogue_answers = get_answers(
                text=dialogue, questions=claims,
                answer_prompt=answer_prompt_faithfulness, **params
            )
            result[f'{col}_faithfulness'] = count_score(claim_answers, dialogue_answers)
            result[f'{col}_faithfulness_summary_answers'] = claim_answers
            result[f'{col}_faithfulness_dialogue_answers'] = dialogue_answers

        for key, value in result.items():
            if isinstance(value, list):
                result[key] = json.dumps(value, ensure_ascii=False)

        rows.append(result)
        pd.DataFrame(rows).to_csv(config['data']['table_path'], index=False)

    return pd.DataFrame(rows)


@click.command()
@click.option('--config', 'config_path', required=True)
def main(config_path):
    load_dotenv()
    config = load_config(config_path)
    data = config['data']
    model = config['model']

    if Path(data['table_path']).resolve() == Path(data['result_path']).resolve():
        raise ValueError("table_path and result_path must be different files")
    for path in [data['table_path'], data['result_path']]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building evaluation table")
    df = make_faithfulness_coverage_df(
        data['test_path'], data['base_llm_path'], data['finetuned_path'],
        data['table_path'], text_col=data['text_col']
    )

    with Groq(max_retries=5) as client:
        logger.info("Generating coverage questions")
        qg = config['coverage']['question_generation']
        messages = make_messages(
            qg['system'], qg['user'], data['table_path'], data['text_col']
        )
        df['coverage_questions'] = get_questions(
            messages=messages, client=client, **model
        )
        if any(not qs for qs in df['coverage_questions']):
            raise ValueError("Judge returned no coverage questions")

        for col in ['base_llm', 'finetuned_llm']:
            logger.info("Generating faithfulness questions for %s", col)
            qg = config['faithfulness']['question_generation']
            messages = make_messages(
                qg['system'], qg['user'], data['table_path'], col
            )
            questions = get_questions(
                messages=messages, client=client, **model
            )
            texts = [t for t in df[col].unique().tolist() if t.strip()]
            questions_by_text = dict(zip(texts, questions, strict=True))
            df[f'{col}_faithfulness_questions'] = df[col].map(
                lambda text: questions_by_text.get(text, [])
            )

        df = evaluate(df, config, client)

    results = []
    for name in ['base_llm', 'finetuned_llm']:
        results.append({
            'name': name,
            'coverage': df[f'{name}_coverage'].mean(),
            'faithfulness': df[f'{name}_faithfulness'].mean(),
            'n_coverage_scored': int(df[f'{name}_coverage'].count()),
            'n_faithfulness_scored': int(df[f'{name}_faithfulness'].count())
        })

    pd.DataFrame(results).to_csv(data['result_path'], index=False)
    logger.info("Mean scores: %s", results)
    logger.info("Results saved to %s", data['result_path'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    main()

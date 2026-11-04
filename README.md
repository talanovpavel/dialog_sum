# Dialogue Summarization
Our goal is to learn how to generate summaries of dialogs from test.csv.
During the course of this project, we explored three approaches:
 
- Extractive baseline
- Unfine-tuned LLM
- Fine-tuned LLM  

## 1) Data:
Columns:
- dialogue (dialogues to summarize)
- summary (created by a human)
- topic (topic of the dialogue)

You can check the EDA results in notebooks/Eda + baseline.ipynb

Preprocessing command:

    uv run python -m scripts.preprocessing --config configs/preprocessing.yaml

## 2) Baseline
Extractive baseline. It's simply the first and last sentences

You can check the metrics in results.csv

    uv run python -m scripts.baseline --config configs/baseline.yaml

## 3) Raw LLM (Qwen2.5-0.5B-Instruct)

    uv run python -m scripts.llm_eval --config configs/base_llm.yaml
    
or you can use **notebooks/baseline_without_finetune.ipynb** instead

## 4) Fine-tuned LLM (Qwen2.5-0.5B-Instruct)
First, we perform LORA fine-tuning on the model, and then we perform the same evaluation as in the previous step.
The metrics are in results/results.csv

Train:

    uv run python -m scripts.lora_train --config configs/lorav1.yaml


Evaluation:

    uv run python -m scripts.llm_eval --config configs/lorav1.yaml

## 5) LLM as judge
I use the Groq API to calculate faithfulness and coverage

I ran out of limits, so there are no results in the repository

    uv run python -m scripts.llm_as_judge --config configs/llm_as_jugde.yaml



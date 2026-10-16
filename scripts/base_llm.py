import torch
import os

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ.pop("VLLM_USE_V1", None)

from vllm import LLM
from vllm import SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dtype="half",
    max_model_len=2048)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

prompts = ...
SamplingParams = ...

outputs = llm.generate(
    ["Summarize this text: Machine learning allows computers to learn patterns from data."],
    SamplingParams(max_tokens=60, temperature=0.0),
)

print(outputs[0].outputs[0].text)
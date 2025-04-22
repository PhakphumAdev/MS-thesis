import os

from groq import Groq
from datasets import load_dataset
import csv
from tqdm import tqdm
def format_prompt(example):
    choices = "\n".join(example["Choice"])
    return f"{example['Question']}\nChoices:\n{choices}"

dataset = load_dataset("pakphum/5csr")
splits = [
    #"test_physical_en", "test_physical_th",
    #"test_psychological_en","test_psychological_th",
    #"test_social_en", "test_social_th",
    #"test_spatial_en", "test_spatial_th",
    #"test_temporal_en", "test_temporal_th",
]
models = [
    "llama-3.3-70b-versatile",
    #"mistral-saba-24b",
    #"qwen-2.5-32b",
    #"deepseek-r1-distill-llama-70b",
]
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def evaluate_split(split_name, model_name, save_dir="results"):
    data = dataset[split_name]
    records = []
    correct = 0

    for i, example in enumerate(tqdm(data, desc=f"{model_name} - {split_name}")):
        prompt = format_prompt(example)
        true_answer = example["Choice"][example["True label Index"]].strip()
        true_answer_clean = true_answer.lower()

        # Get model response
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert multiple-choice reasoning assistant. For every question, you will be given a set of answer choices. Your task is to read the question carefully and reply with the exact correct answer from the provided choices. Do not explain or rephrase. Respond with only one of the provided choices as it appears."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
        )

        output = response.choices[0].message.content.strip()
        output_clean = output.lower()
        is_correct = output_clean == true_answer_clean
        if is_correct:
            correct += 1

        records.append({
            "index": i,
            "model": model_name,
            "split": split_name,
            "prompt": prompt,
            "true_answer": true_answer,
            "model_output": output,
            "is_correct": is_correct
        })

    # Save CSV
    model_dir = os.path.join(save_dir, model_name.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, f"{split_name}.csv")
    
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    accuracy = correct / len(data)
    return accuracy

results = {}

for model in models:
    results[model] = {}
    for split in splits:
        print(f"Evaluating {model} on {split}...")
        acc = evaluate_split(split, model)
        results[model][split] = acc
        print(f"{split}: {acc:.2%}")
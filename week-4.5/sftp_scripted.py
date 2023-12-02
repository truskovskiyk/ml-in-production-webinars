# Based on https://colab.research.google.com/drive/16LXuIcps14M8xmOOZ_FCPaH7SKJuUlE1#scrollTo=bmERa50SayQl
import torch
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

# peft module helps us generate & inject LoRA modules into base model
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

# transformers module helps us load a base model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig, # helper to quantize the model so we can run on a single GPU
    TrainingArguments,
)

# trl modules help us train LoRA weights
from trl import SFTTrainer


# huggingface datasets module
import datasets
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")


TRAINING_CLASSIFIER_PROMPT_v2 = """### Sentence:{sentence} ### Class:{label}"""
INFERENCE_CLASSIFIER_PROMPT_v2 = """### Sentence:{sentence} ### Class:"""

def get_newsgroup_instruction_data(mode, texts, labels):
    # this function injects the prompt above to the dataset
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text,
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                sentence=text,
            )
        instructions.append(example)

    return instructions


def clean_newsgroup_data(texts, labels):
    label2data = {}
    clean_data, clean_labels = [], []
    for data, label in zip(texts, labels):
        if isinstance(data, str) and isinstance(label, str):
            clean_data.append(data)
            clean_labels.append(label)

            if label not in label2data:
                label2data[label] = data

    return label2data, clean_data, clean_labels


def get_newsgroup_data_for_ft(mode="train", train_sample_fraction=0.99):
    newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    train_data = newsgroup_dataset["train"]["text"]
    train_labels = newsgroup_dataset["train"]["label"]
    label2data, train_data, train_labels = clean_newsgroup_data(
        train_data, train_labels
    )

    test_data = newsgroup_dataset["test"]["text"]
    test_labels = newsgroup_dataset["test"]["label"]
    _, test_data, test_labels = clean_newsgroup_data(test_data, test_labels)

    # sample n points from training data
    train_df = pd.DataFrame(data={"text": train_data, "label": train_labels})
    train_df, _ = train_test_split(
        train_df,
        train_size=train_sample_fraction,
        stratify=train_df["label"],
        random_state=42,
    )
    train_data = train_df["text"]
    train_labels = train_df["label"]

    train_instructions = get_newsgroup_instruction_data(mode, train_data, train_labels)
    test_instructions = get_newsgroup_instruction_data(mode, test_data, test_labels)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )

    return train_dataset, test_dataset


def get_newsgroup_classes():
    newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    train_data = newsgroup_dataset["train"]["text"]
    train_labels = newsgroup_dataset["train"]["label"]

    label2data, clean_data, clean_labels = clean_newsgroup_data(
        train_data, train_labels
    )
    df = pd.DataFrame(data={"text": clean_data, "label": clean_labels})

    newsgroup_classes = df["label"].unique()
    newsgroup_classes = ", ".join(newsgroup_classes)

    return newsgroup_classes

sample_fraction = 0.25 # editable

train_dataset, _ = get_newsgroup_data_for_ft(mode="train", train_sample_fraction=sample_fraction)
_, test_dataset = get_newsgroup_data_for_ft(mode="inference")
newsgroup_classes = get_newsgroup_classes()

print(f"Sample fraction:{sample_fraction}")
print(f"Training samples:{train_dataset.shape}")

train_dataset['instructions'][0]
test_dataset['instructions'][0]


# Load model and tokenizer
pretrained_ckpt = "NousResearch/Llama-2-7b-hf"
# pretrained_ckpt = "mistralai/Mistral-7B-v0.1"

# BitsAndBytesConfig quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_compute_type=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_ckpt,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto",
)
model.config.pretraining_tp = 1 #value different than 1 will activate the more accurate but slower computation of the linear layers

tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def infer_one_example(model, tokenizer, instruction):
  input_ids = tokenizer(instruction, return_tensors="pt", truncation=True).input_ids.cuda()

  with torch.inference_mode():
    try:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.95,
            temperature=1e-3,
        )
        result = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
        result = result[len(instruction) :]

    except:
        # oops, it's too long!
        result = ""

  return result


instruction, label = test_dataset["instructions"][0], test_dataset["labels"][0]
instruction, label


ZERO_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas:

{newsgroup_classes}

From the above list of classes, select only one class that the provided sentence can be classified into. The sentence will be delimited with triple backticks. Once again, only predict the class from the given list of classes. Do not predict anything else.

### Sentence: ```{sentence}```
### Class:
"""

# inject data into the prompt
prompt_zeroshot = ZERO_SHOT_CLASSIFIER_PROMPT.format(
    newsgroup_classes = newsgroup_classes,
    sentence = instruction
)

infer_one_example(model, tokenizer, prompt_zeroshot)

dropout = 0.1
epochs = 3    # 3 epochs takes ~20 min on a T4

# LoRA Configs
rank = 16      # try larger value for more complex task
alpha = 32    # try larger value if task is substantially different from language understanding/processing


# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=alpha,
    lora_dropout=dropout,
    r=rank,
    bias="none",
    task_type="CAUSAL_LM"
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

results_dir = "finetuned_model"


training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=epochs,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=2e-4,
        # bf16=True, # Set to true if you're using A10/A100
        # tf32=True, # Set to true if you're using A10/A100
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
    )

max_seq_length = 512  # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    args=training_args,
    dataset_text_field="instructions",
)

trainer_stats = trainer.train()
train_loss = trainer_stats.training_loss
print(f"Training loss:{train_loss}")

peft_model_id = f"{results_dir}/assets"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

with open(f"{results_dir}/results.pkl", "wb") as handle:
    run_result = [
        epochs,
        rank,
        dropout,
        train_loss,
    ]
    pickle.dump(run_result, handle)
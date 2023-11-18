import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
import datasets
from datasets import Dataset, load_dataset

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from prompts import TRAINING_SUMMARIZATION_PROMPT_v2


def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = TRAINING_SUMMARIZATION_PROMPT_v2

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue=dialogue,
            summary=summary,
        )
        instructions.append(example)

    return instructions


def prepare_samsum_data():
    dataset = load_dataset("samsum")
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    dialogues = train_dataset["dialogue"]
    summaries = train_dataset["summary"]
    train_instructions = prepare_instructions(dialogues, summaries)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions})
    )

    return train_dataset


def main(args):
    train_dataset = prepare_samsum_data()

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    results_dir = f"experiments/summarization_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}"

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        # disable_tqdm=True # disable tqdm since with packing values are in correct
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
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


import argparse
import torch
import os
import pandas as pd
import evaluate
from datasets import load_dataset
import pickle
import warnings

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from prompts import INFERENCE_SUMMARIZATION_PROMPT_v2

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = INFERENCE_SUMMARIZATION_PROMPT_v2

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue=dialogue,
        )
        instructions.append(example)

    return instructions


def prepare_samsum_data():
    dataset = load_dataset("samsum")
    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    summaries = val_dataset["summary"]
    val_instructions = prepare_instructions(dialogues, summaries)

    return val_instructions, summaries


def main(args):
    val_instructions, summaries = prepare_samsum_data()

    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/assets"

    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    results = []
    for instruct, summary in zip(val_instructions, summaries):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=1e-2,
            )
            result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0]
            result = result[len(instruct) :]
            results.append(result)
            print(f"Instruction:{instruct}")
            print(f"Summary:{summary}")
            print(f"Generated:{result}")
            print("----------------------------------------")

    # compute metric
    rouge = metric.compute(predictions=results, references=summaries, use_stemmer=True)

    metrics = {metric: round(rouge[metric] * 100, 2) for metric in rouge.keys()}

    save_dir = os.path.join(experiment, "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        default="experiments/summarization_epochs-1_rank-64_dropout-0.1",
    )

    args = parser.parse_args()
    main(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    args = parser.parse_args()
    main(args)


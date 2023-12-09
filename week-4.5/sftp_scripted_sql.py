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


TRAINING_SUMMARIZATION_PROMPT_v2 = """### Context:{context} ### Auestion:{question} ### Answer:{answer}"""
INFERENCE_SUMMARIZATION_PROMPT_v2 = """### Context:{context} ### Auestion:{question} ### Answer:"""

INFERENCE_SUMMARIZATION_PROMPT = """Write SQL query based on.

### Context: ```{context}```
### Question: ```{question}```
### Answer: 
"""


def prepare_instructions(context, question, answer):
    instructions = []

    prompt = TRAINING_SUMMARIZATION_PROMPT_v2

    for c, q, a in zip(context, question, answer):
        example = prompt.format(
            context=c, 
            question=q, 
            answer=a,
        )
        instructions.append(example)

    return instructions


def prepare_sql_data():
    dataset = load_dataset("b-mc2/sql-create-context")
    train_dataset = dataset["train"]

    context = train_dataset["context"]
    question = train_dataset["question"]
    answer = train_dataset["answer"]

    train_instructions = prepare_instructions(context, question, answer)
    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_instructions, 'context': context, "question": question, "answer": answer})
    )

    return train_dataset

def infer_one_example(model, tokenizer, instruction):
  input_ids = tokenizer(instruction, return_tensors="pt", truncation=True).input_ids.cuda()

  with torch.inference_mode():
    try:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
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

def main():
    train_dataset = prepare_sql_data()
    print(train_dataset['instructions'][0])

    pretrained_ckpt = "NousResearch/Llama-2-7b-hf"
    pretrained_ckpt = "finetuned_model_sql/assets/"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    
    dropout = 0.1
    epochs = 3    # 3 epochs takes ~20 min on a T4

    # LoRA Configs
    rank = 16      # try larger value for more complex task
    alpha = 32    # try larger value if task is substantially different from language understanding/processing


    # LoRA config based on QLoRA paper
    # peft_config = LoraConfig(
    #     lora_alpha=alpha,
    #     lora_dropout=dropout,
    #     r=rank,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )


    # # prepare model for training
    # model = prepare_model_for_kbit_training(model)
    # model = get_peft_model(model, peft_config)


    # inject data into the prompt
    prompt_zeroshot = INFERENCE_SUMMARIZATION_PROMPT.format(
        context = train_dataset['context'][0],
        question = train_dataset['question'][0],
    )

    infer_one_example(model, tokenizer, prompt_zeroshot)



    results_dir = "finetuned_model_sql_test"

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=epochs,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
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



if __name__ == "__main__":
    main()
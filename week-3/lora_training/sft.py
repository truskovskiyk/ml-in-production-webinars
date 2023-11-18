from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from trl import SFTTrainer, is_xpu_available
from pprint import pprint

tqdm.pandas()




@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="instructions", metadata={"help": "the text field of the dataset"})

    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=4, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Whether to use gradient checkpointing or no"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

# Step 2: Load the dataset
from lora_training.datasets_prep import get_newsgroup_data_for_ft
train_dataset, _ = get_newsgroup_data_for_ft(mode="train", train_sample_fraction=0.99)
# dataset = load_dataset(script_args.dataset_name, split="train")
# dataset = load_dataset("glue", "cola", split="train")
# dataset = dataset.shuffle()
# dataset = dataset.select(range(1000))
# example = dataset[:3]

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['sentence'])):
#         result = "Correct" if example['label'][i] == 1 else "Incorrect"
#         text = f"### Sentence: {example['sentence'][i]}\n ### Result: {result}"
#         output_texts.append(text)
#     return output_texts

# formatting_prompts_func(example=example)

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    gradient_checkpointing=script_args.gradient_checkpointing,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None




# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=train_dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    packing=True,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)


# instruct = '### Sentence: John is shorter than five feet.\n ### Result:'

# tokenizer = trainer.tokenizer
# input_ids = tokenizer(instruct, return_tensors="pt", truncation=True).input_ids.cuda()


# outputs = model.generate(input_ids=input_ids, max_new_tokens=20, do_sample=True, top_p=0.95, temperature=1e-3)
# result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
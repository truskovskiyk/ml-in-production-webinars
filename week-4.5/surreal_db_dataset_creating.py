import typer 


def map_sql_create_context_to_surreal_ql():


def get_sql_data_paired():

    dataset = load_dataset("b-mc2/sql-create-context")
    train_dataset = dataset["train"]
    original_columns = train_dataset.column_names

    def return_prompt_and_responses(samples):
        return {
            "prompt": [INFERENCE_SUMMARIZATION_PROMPT_v2.format(context=context, question=question) for context, question in zip(samples["context"], samples['question'])],
            "chosen": [x for x in samples["answer"]],
            # "rejected": samples["answer_rejected"],
            "rejected": ["Random stuff" for x in samples["answer"]],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )    
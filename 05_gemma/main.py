import json
import os
import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.lora import train, TrainingArgs, train_model, run
from datasets import load_dataset
import types
from transformers import AutoTokenizer


# Global variables to store the models and tokenizers
model1 = None
tokenizer1 = None
model2 = None
tokenizer2 = None


class Args:
    def __init__(
        self,
        batch_size,
        iters,
        lora_layers,
        max_seq_length,
        learning_rate,
        adapter_path,
        resume_adapter_file,
        grad_checkpoint,
        lora_parameters,
        steps_per_report=10,
        steps_per_eval=200,
        save_every=100,
        val_batches=25,
        lr_schedule=None,
        use_dora=False,
    ):
        self.batch_size = batch_size
        self.iters = iters
        self.lora_layers = lora_layers
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.adapter_path = adapter_path
        self.resume_adapter_file = resume_adapter_file
        self.grad_checkpoint = grad_checkpoint
        self.lora_parameters = lora_parameters
        self.steps_per_report = steps_per_report
        self.steps_per_eval = steps_per_eval
        self.save_every = save_every
        self.val_batches = val_batches
        self.lr_schedule = lr_schedule
        self.use_dora = use_dora


def calculate_training_parameters(
    batch_size,
    epochs,
    train_set,
    lora_layers,
    context_length,
    learning_rate,
    model_path,
):
    try:
        # Calculate the number of training steps in an epoch
        train_file = os.path.join(train_set, "train.jsonl")
        iters_per_epoch = -(-sum(1 for _ in open(train_file)) // batch_size)
        iters = iters_per_epoch * epochs

        model, tokenizer = load(model_path)

        # Function to count tokens in the file
        def count_tokens_in_file(token_model, file_path):
            total_tokens = 0
            with open(file_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    tokens = token_model.tokenize(text)
                    total_tokens += len(tokens)
            return total_tokens

        total_tokens = count_tokens_in_file(tokenizer, train_file) * epochs
        multiplier_for_layers = 1 / (1 + (((32 - lora_layers) / 32) * 1.5))
        training_rate = 500 // multiplier_for_layers
        estimated_total_time = int(total_tokens // training_rate)
        estimated_minutes = int(estimated_total_time // 60)
        estimated_seconds = int(estimated_total_time % 60)
        slow_time = estimated_total_time * 7
        slow_minutes = int(slow_time // 60)
        slow_seconds = int(slow_time % 60)

        result = (
            f"\nAutomatically detected {iters_per_epoch} data entries."
            f"\nFor {epochs} epoch(s) with a batch size of {batch_size}, we will set iters to: {iters}"
            f"\nTotal number of tokens in the JSONL file: {total_tokens}"
            f"\nEstimated training rate in tokens/second if fits in GPU: {training_rate}"
            f"\n\nIf model fits in GPU: Estimated time for {epochs} epoch(s) with {lora_layers} LoRA layer(s) with a token amount of {total_tokens}: "
            f"\n{estimated_minutes} minutes and {estimated_seconds} seconds"
            f"\n\nElse if model doesn't fit in GPU, could be up to:"
            f"\n{slow_minutes} minutes and {slow_seconds} seconds\n"
        )
        return result
    except Exception as e:
        return str(e)


def start_training_process(
    batch_size,
    epochs,
    train_set,
    lora_layers,
    context_length,
    learning_rate,
    model_path,
):
    try:
        # Calculate iterations
        train_file = os.path.join(train_set, "train.jsonl")
        iters_per_epoch = -(-sum(1 for _ in open(train_file)) // batch_size)
        iters = iters_per_epoch * epochs

        # Define Args class
        class Args:
            def __init__(
                self,
                model,
                train,
                data,
                lora_layers,
                batch_size,
                iters,
                learning_rate,
                steps_per_report,
                steps_per_eval,
                save_every,
                max_seq_length,
                adapter_path,
                lora_parameters,
                resume_adapter_file=None,
                grad_checkpoint=False,
                val_batches=25,
                lr_schedule=None,
                use_dora=False,
                seed=0,
                test=False,
                test_batches=500,
            ):
                self.model = model
                self.train = train
                self.data = data
                self.lora_layers = lora_layers
                self.batch_size = batch_size
                self.iters = iters
                self.learning_rate = learning_rate
                self.steps_per_report = steps_per_report
                self.steps_per_eval = steps_per_eval
                self.save_every = save_every
                self.max_seq_length = max_seq_length
                self.adapter_path = adapter_path
                self.lora_parameters = lora_parameters
                self.resume_adapter_file = resume_adapter_file
                self.grad_checkpoint = grad_checkpoint
                self.val_batches = val_batches
                self.lr_schedule = lr_schedule
                self.use_dora = use_dora
                self.seed = seed
                self.test = test
                self.test_batches = test_batches

        # Create Args instance
        args = Args(
            model=model_path,
            train=True,
            data=train_set,
            lora_layers=-1,
            batch_size=batch_size,
            iters=iters,
            learning_rate=learning_rate,
            steps_per_report=5,
            steps_per_eval=100,
            save_every=50,
            max_seq_length=context_length,
            adapter_path="./adapters",
            lora_parameters={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
            resume_adapter_file=None,
            grad_checkpoint=False,
        )

        # Run the training process
        run(types.SimpleNamespace(**vars(args)))

        return "Model training complete."
    except Exception as e:
        return str(e)


def start_training_process_old(
    batch_size,
    epochs,
    train_set,
    lora_layers,
    context_length,
    learning_rate,
    model_path,
):
    try:
        # Load model and tokenizer
        print(f"\nTraining model...\n")
        print(model_path)

        # Load model and tokenizer
        model, tokenizer = load(model_path)

        # Prepare datasets
        train_dataset = load_dataset(
            "json", data_files=os.path.join(train_set, "train.jsonl")
        )["train"]
        val_dataset = load_dataset(
            "json", data_files=os.path.join(train_set, "valid.jsonl")
        )["train"]

        train_file = os.path.join(train_set, "train.jsonl")
        iters_per_epoch = -(-sum(1 for _ in open(train_file)) // batch_size)
        iters = iters_per_epoch * epochs

        # Training arguments
        training_args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            max_seq_length=context_length,
            steps_per_save=100,
            steps_per_eval=500,
            steps_per_report=10,
        )

        # Dummy optimizer
        from torch.optim import AdamW

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Train the model
        train(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=training_args,
        )

        return "Model training complete."
    except Exception as e:
        return str(e)


def load_and_process_dataset(dataset_name, train_percentage, eos_token=""):
    try:
        # Load the full dataset
        full_dataset = load_dataset(dataset_name, split="train")
        total_size = len(full_dataset)

        # Calculate split sizes
        train_size = int(total_size * (train_percentage / 100))
        valid_size = int(total_size * ((100 - train_percentage) / 100))

        # Create the train and validation splits
        dataset = full_dataset.select(range(train_size))
        valid_dataset = full_dataset.select(range(train_size, train_size + valid_size))

        data_dir = "./data/datasets"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        modified_train_file = f"{data_dir}/train.jsonl"
        valid_file = f"{data_dir}/valid.jsonl"

        # Save the processed datasets with EOS token
        with open(modified_train_file, "w") as train_f:
            for item in dataset:
                item_with_eos = item["text"] + eos_token
                train_f.write(json.dumps({"text": item_with_eos}) + "\n")

        with open(valid_file, "w") as valid_f:
            for item in valid_dataset:
                item_with_eos = item["text"] + eos_token
                valid_f.write(json.dumps({"text": item_with_eos}) + "\n")

        # Preview the first few entries
        preview = "\nPreview:\n"
        with open(modified_train_file, "r") as file:
            for i, line in enumerate(file):
                if i >= 5:
                    break
                entry = json.loads(line)
                preview += "=========================\n"
                preview += entry["text"] + "\n"

        return (
            f"Datasets loaded and processed.\nTrain file: {modified_train_file}\nValid file: {valid_file}\n"
            + preview
        )
    except Exception as e:
        return str(e)


def generate_text_completion(model_path, adapter_path, prompt, max_tokens):
    if not model_path or not adapter_path:
        return "Please provide valid model paths", "Please provide valid model paths"

    global model1, tokenizer1, model2, tokenizer2
    if not model1 or not tokenizer1:
        model1, tokenizer1 = load(model_path)

    if not os.path.exists(adapter_path):
        result1 = ""
        for text_segment in stream_generate(
            model1, tokenizer1, prompt, max_tokens=max_tokens
        ):
            result1 += text_segment
            yield result1, ""
    else:
        if not model2 or not tokenizer2:
            model2, tokenizer2 = load(model_path, adapter_path=adapter_path)

        result1 = ""
        result2 = ""

        for text_segment in stream_generate(
            model1, tokenizer1, prompt, max_tokens=max_tokens
        ):
            result1 += text_segment
            yield result1, result2

        for text_segment in stream_generate(
            model2, tokenizer2, prompt, max_tokens=max_tokens
        ):
            result2 += text_segment
            yield result1, result2


def clear_controls(response1, response2):
    response1.value = ""
    response2.value = ""
    return response1, response2


with gr.Blocks(theme="gradio/monochrome") as demo:
    with gr.Tabs():
        with gr.Tab("Load Dataset"):
            with gr.Row():
                with gr.Column():
                    dataset_name = gr.Textbox(
                        "cyan2k/promptvieh_dataset", label="Hugging Face Dataset Name"
                    )
                    train_percentage = gr.Slider(
                        0, 100, value=80, step=1, label="Train Percentage"
                    )
                    load_button = gr.Button("Load and Process Dataset")
                with gr.Column():
                    dataset_output = gr.Textbox(
                        label="Dataset Output", lines=10, interactive=True
                    )

            load_button.click(
                fn=load_and_process_dataset,
                inputs=[dataset_name, train_percentage],
                outputs=dataset_output,
            )

        with gr.Tab("Training Parameters"):
            with gr.Row():
                with gr.Column():
                    batch_size = gr.Number(value=2, label="Batch Size")
                    epochs = gr.Number(value=1, label="Epochs")
                    train_set = gr.Textbox(
                        value="./data/datasets", label="Train Set Path"
                    )
                    lora_layers = gr.Number(value=22, label="LoRA Layers")
                    context_length = gr.Number(value=512, label="Context Length")
                    learning_rate = gr.Number(value=6e-5, label="Learning Rate")
                    model_path = gr.Textbox(value="./model/gemma2", label="Model Path")
                    calculate_button = gr.Button("Calculate Training Parameters")
                    start_training = gr.Button("Start Training")
                with gr.Column():
                    calculation_output = gr.Textbox(
                        label="Training Output", lines=20, interactive=True
                    )
                    training_output = gr.Textbox(
                        label="Training Output", lines=20, interactive=True
                    )

            calculate_button.click(
                fn=calculate_training_parameters,
                inputs=[
                    batch_size,
                    epochs,
                    train_set,
                    lora_layers,
                    context_length,
                    learning_rate,
                    model_path,
                ],
                outputs=calculation_output,
            )
            start_training.click(
                fn=start_training_process,
                inputs=[
                    batch_size,
                    epochs,
                    train_set,
                    lora_layers,
                    context_length,
                    learning_rate,
                    model_path,
                ],
                outputs=training_output,
            )

        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column():
                    model_path1 = gr.Textbox("model/gemma2", label="Model Path")
                    adapter_path = gr.Textbox("./adapters", label="Adapter Path")
                    max_tokens = gr.Slider(1, 1000, value=100, label="Max Tokens")
                    prompt = gr.Textbox("Once upon a time", label="Prompt")
                    generate_button = gr.Button("Generate Text Completion")
                with gr.Column():
                    response1 = gr.Textbox(
                        label="Original Model", lines=10, interactive=True
                    )
                    response2 = gr.Textbox(
                        label="finetuned Model", lines=10, interactive=True
                    )

            prompt.submit(
                fn=clear_controls,
                inputs=[response1, response2],
                outputs=[response1, response2],
                queue=False,
            ).then(
                fn=generate_text_completion,
                inputs=[model_path1, adapter_path, prompt, max_tokens],
                outputs=[response1, response2],
            )

            generate_button.click(
                fn=generate_text_completion,
                inputs=[model_path1, adapter_path, prompt, max_tokens],
                outputs=[response1, response2],
            )

demo.queue()
demo.launch(share=True)

import json
import os
import gradio as gr
from mlx_lm import load, stream_generate
from datasets import load_dataset

# Global variables to store the models and tokenizers
model1 = None
tokenizer1 = None
model2 = None
tokenizer2 = None


def load_and_process_dataset(dataset_name, split_train, split_valid):
    try:
        dataset = load_dataset(dataset_name, split=split_train)
        valid_dataset = load_dataset(dataset_name, split=split_valid)
        print(f"\nDatasets loaded.\n")

        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        modified_train_file = f"{data_dir}/train.jsonl"
        valid_file = f"{data_dir}/valid.jsonl"

        # Assuming some processing here
        with open(modified_train_file, "w") as train_f:
            for example in dataset:
                train_f.write(json.dumps(example) + "\n")

        with open(valid_file, "w") as valid_f:
            for example in valid_dataset:
                valid_f.write(json.dumps(example) + "\n")

        return f"Datasets loaded and processed.\nTrain file: {modified_train_file}\nValid file: {valid_file}"
    except Exception as e:
        return str(e)


def generate_text_completion(model_path1, model_path2, prompt, max_tokens):
    if not model_path1 or not model_path2:
        return "Please provide valid model paths", "Please provide valid model paths"

    global model1, tokenizer1, model2, tokenizer2
    if not model1 or not tokenizer1:
        model1, tokenizer1 = load(model_path1)
    if not model2 or not tokenizer2:
        model2, tokenizer2 = load(model_path2)

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
                    split_train = gr.Textbox("train[:80000]", label="Train Split")
                    split_valid = gr.Textbox("train[80001:]", label="Valid Split")
                    load_button = gr.Button("Load and Process Dataset")
                with gr.Column():
                    dataset_output = gr.Textbox(
                        label="Dataset Output", lines=10, interactive=True
                    )

            load_button.click(
                fn=load_and_process_dataset,
                inputs=[dataset_name, split_train, split_valid],
                outputs=dataset_output,
            )

        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column():
                    model_path1 = gr.Textbox("data/gemma2", label="Model Path 1")
                    model_path2 = gr.Textbox("data/gemma2", label="Model Path 2")
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
                inputs=[model_path1, model_path2, prompt, max_tokens],
                outputs=[response1, response2],
            )

            generate_button.click(
                fn=generate_text_completion,
                inputs=[model_path1, model_path2, prompt, max_tokens],
                outputs=[response1, response2],
            )

demo.queue()
demo.launch(share=True)

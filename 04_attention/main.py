import gradio as gr
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt


# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)


def visualize_attention(sentence):
    # Tokenize and prepare input
    inputs = tokenizer.encode_plus(
        sentence, return_tensors="pt", add_special_tokens=True
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Pass through model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get attention weights from the first layer and the first head
    attention = outputs.attentions[0][0][0].detach().numpy()

    # Create attention heatmap
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    plt.figure(figsize=(12, 10))
    plt.imshow(attention, cmap="viridis", interpolation="nearest")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.colorbar()
    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.show()


# Create Gradio interface
iface = gr.Interface(
    fn=visualize_attention,
    inputs=gr.Textbox(label="Enter a sentence:"),
    outputs="plot",
)

iface.launch()

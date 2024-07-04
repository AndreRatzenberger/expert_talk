import re
import gradio as gr
from collections import Counter
import nltk
from nltk.util import ngrams
import pandas as pd

nltk.download("punkt")

# Mapping of n-gram types to integers
ngram_mapping = {"Unigram": 1, "Bigram": 2, "Trigram": 3}


def preprocess_text(text):
    # Remove non-letter characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text


def generate_ngrams(text, ngram_type):
    # Preprocess the text
    text = preprocess_text(text)

    # Tokenize the input text
    tokens = nltk.word_tokenize(text)

    # Get the n value from the ngram type
    n = ngram_mapping[ngram_type]

    # Generate n-grams
    n_grams = list(ngrams(tokens, n))

    # Count the frequency of each n-gram
    ngram_freq = Counter(n_grams)
    total_ngrams = sum(ngram_freq.values())

    # Prepare the results for display in a table format
    ngram_data = [
        {
            "N-gram": " ".join(gram),
            "Frequency": count,
            "Percentage": f"{(count / total_ngrams) * 100:.2f}%",
        }
        for gram, count in ngram_freq.items()
    ]
    ngram_df = pd.DataFrame(ngram_data)

    return ngram_df


def process_input(text, file_path, ngram_type):
    if file_path is not None:
        with open(file_path, "r") as file:
            text = file.read()
    elif not text:
        return pd.DataFrame(columns=["N-gram", "Frequency"])
    return generate_ngrams(text, ngram_type)


# Define the Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(lines=5, label="Enter text"),
        gr.File(label="Upload a .txt file"),
        gr.Dropdown(
            choices=["Unigram", "Bigram", "Trigram"],
            label="Select N-gram Type",
            value="Unigram",
        ),
    ],
    outputs=gr.Dataframe(headers=["N-gram", "Frequency"], label="N-gram Frequencies"),
    title="N-gram Model Demonstration",
    description="Input a sentence or paragraph, upload a .txt file, select the type of n-gram (unigram, bigram, trigram), and see the n-gram frequencies in a table.",
    theme="gradio/monochrome",
)

# Launch the app
iface.launch()

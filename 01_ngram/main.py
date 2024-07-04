import random
import re
import gradio as gr
from collections import Counter
import nltk
from nltk.util import ngrams
import pandas as pd
from fastai.text.all import *

nltk.download("punkt")

# Mapping of n-gram types to integers
ngram_mapping = {"Unigram": 1, "Bigram": 2, "Trigram": 3, "4-gram": 4}
ngram_freq = {}
n = 0


def preprocess_text(text):
    # Remove non-letter characters except for German umlauts and convert to lowercase
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", "", text)
    text = text.lower()
    return text


def generate_ngrams(text, ngram_type):
    global n, ngram_freq
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


def generate_text(num_words):
    global ngram_freq, n
    if not ngram_freq:
        return "N-gram data is not available. Please generate n-grams first."
    # Start with a random n-gram
    current_ngram = random.choice(list(ngram_freq.keys()))
    generated_text = list(current_ngram)

    for _ in range(num_words - n):
        # Get the possible next words
        next_words = [gram[-1] for gram in ngram_freq if gram[:-1] == current_ngram[1:]]

        if next_words:
            next_word = random.choice(next_words)
            generated_text.append(next_word)
            current_ngram = tuple(generated_text[-n:])
        else:
            break

    text = " ".join(generated_text)
    return text


def process_input(text, file_path, ngram_type):
    if file_path is not None:
        with open(file_path, "r") as file:
            text = file.read()
    elif not text:
        return pd.DataFrame(columns=["N-gram", "Frequency"])
    return generate_ngrams(text, ngram_type)


def generate_text_with_rnn(seed_text, num_words):
    data_lm = TextDataLoaders.from_folder("data", valid_pct=0.2, is_lm=True, seq_len=72)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    learn.fit_one_cycle(10, 1e-2)
    learn.save_encoder("fine_tuned_enc")


# Define the Gradio interface
tab1 = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(lines=5, label="Enter text"),
        gr.File(label="Upload a .txt file"),
        gr.Dropdown(
            choices=["Unigram", "Bigram", "Trigram", "4-gram"],
            label="Select N-gram Type",
            value="Unigram",
        ),
    ],
    outputs=gr.Dataframe(headers=["N-gram", "Frequency"], label="N-gram Frequencies"),
    title="Generate n-grams from Text",
    description="Input a sentence or paragraph, upload a .txt file, select the type of n-gram (unigram, bigram, trigram), and see the n-gram frequencies in a table.",
)

# Launch the app
# Define the Gradio interface for Tab 2
tab2 = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Slider(minimum=10, maximum=100, step=1, label="Text Length"),
    ],
    outputs="text",
    title="Text Generation Using N-grams",
    description="Generate text based on the n-grams generated in Tab 1. Choose the length of the generated text.",
    theme="gradio/monochrome",
)

tab3 = gr.Interface(
    fn=generate_text_with_rnn,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter starting text", label="Starting Text"),
        gr.Slider(
            minimum=1, maximum=50, step=1, label="Number of Words to Generate", value=10
        ),
    ],
    outputs="text",
    title="RNN Text Generation with Fastai",
    description="Input a starting text and the number of words to generate. The app uses a Fastai RNN model to generate the next sequence of words.",
)

# Define the tabbed layout
app = gr.TabbedInterface(
    [tab1, tab2, tab3],
    title="N-gram Model Demonstration",
    tab_names=["N-gram Model", "Text Generation", "RNN Text Generation"],
    theme="gradio/monochrome",
)

# Launch the app
app.launch()

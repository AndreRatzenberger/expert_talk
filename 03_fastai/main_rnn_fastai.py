import gradio as gr
from fastai.text.all import *
import gradio as gr

# Sample corpus for training the RNN model
corpus = [
    "hello world",
    "hello there",
    "hello world there",
    "world of text generation",
    "text generation using rnn",
    "simple rnn model",
    "demonstrate text generation",
    "using rnn for text generation",
]


# Load the text data
dls = TextDataLoaders.from_folder("data/", valid_pct=0.2, is_lm=True, seq_len=72)

# Build the RNN model using Fastai
learn = language_model_learner(dls, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(100, 1e-2)

# Save the encoder for text generation
learn.save_encoder("fine_tuned_enc")


# Function to generate text
def generate_text(seed_text, next_words):
    learn.load_encoder("fine_tuned_enc")
    generated_text = learn.predict(seed_text, n_words=next_words)
    return generated_text


# Define the Gradio interface
iface = gr.Interface(
    fn=generate_text,
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

# Launch the app
iface.launch()

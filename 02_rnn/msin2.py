import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import random

# Preprocess the corpus
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


def preprocess_text(text):
    # Remove non-letter characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text


corpus = [preprocess_text(line) for line in corpus]
all_text = " ".join(corpus)
words = list(set(all_text.split()))
word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for idx, word in enumerate(words)}

# Create sequences
sequence_length = 4
sequences = []
for line in corpus:
    tokens = line.split()
    for i in range(len(tokens) - sequence_length):
        sequences.append(tokens[i : i + sequence_length + 1])


class TextDataset(Dataset):
    def __init__(self, sequences, word2idx):
        self.sequences = sequences
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = torch.tensor(
            [self.word2idx[word] for word in sequence[:-1]], dtype=torch.long
        )
        target_seq = torch.tensor(self.word2idx[sequence[-1]], dtype=torch.long)
        return input_seq, target_seq


dataset = TextDataset(sequences, word2idx)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, hidden_dim)


vocab_size = len(word2idx)
embedding_dim = 10
hidden_dim = 50
model = RNNModel(vocab_size, embedding_dim, hidden_dim)

# Training the model
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    hidden = model.init_hidden(batch_size=1)
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")


# Function to generate text
def generate_text(seed_text, next_words):
    model.eval()
    words = seed_text.split()
    hidden = model.init_hidden(batch_size=1)
    for _ in range(next_words):
        input_seq = torch.tensor(
            [[word2idx[word] for word in words[-sequence_length:]]], dtype=torch.long
        )
        output, hidden = model(input_seq, hidden)
        predicted_word_idx = output.argmax(dim=1).item()
        predicted_word = idx2word[predicted_word_idx]
        words.append(predicted_word)
    return " ".join(words)


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
    title="RNN Text Generation with PyTorch",
    description="Input a starting text and the number of words to generate. The app uses a PyTorch RNN model to generate the next sequence of words.",
)

# Launch the app
iface.launch()

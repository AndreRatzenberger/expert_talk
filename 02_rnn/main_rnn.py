import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import matplotlib.pyplot as plt
import wandb

wandb.login()


# parameters
is_training_done = False
embedding_dim = 10
hidden_dim = 50
sequence_length = 4
num_epochs = 30
learning_rate = 0.001

# global variables
sequences = []
corpus = []
model = None
word2idx = None
idx2word = None
loss_plot = None

run = wandb.init(
    # Set the project where this run will be logged
    project="wd-expert-talk",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
    },
)


# Preprocess the text
def preprocess_text(text):
    # Remove non-letter characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text


# Function to handle file input
def handle_file_upload(file):
    lines = open(file).readlines()
    # lines = text.splitlines()
    corpus = [preprocess_text(line) for line in lines if line.strip()]
    return corpus


# Create sequences
sequence_length = 4


def create_sequences(corpus):
    all_text = " ".join(corpus)
    words = list(set(all_text.split()))
    word2idx = {word: idx for idx, word in enumerate(words)}
    word2idx["<UNK>"] = len(word2idx)  # Adding a special token for unknown words
    idx2word = {idx: word for idx, word in enumerate(words)}
    idx2word[len(word2idx) - 1] = "<UNK>"

    sequences = []
    for line in corpus:
        tokens = line.split()
        for i in range(len(tokens) - sequence_length):
            sequences.append(tokens[i : i + sequence_length + 1])
    return word2idx, idx2word, sequences


class TextDataset(Dataset):
    def __init__(self, sequences, word2idx):
        self.sequences = sequences
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = torch.tensor(
            [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sequence[:-1]],
            dtype=torch.long,
        )
        target_seq = torch.tensor(
            self.word2idx.get(sequence[-1], self.word2idx["<UNK>"]), dtype=torch.long
        )
        return input_seq, target_seq


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
        return torch.zeros(1, batch_size, hidden_dim, dtype=torch.float)


# Function to train the model
def train_model(
    corpus,
    number_epochs,
    learning_rate,
    embedding_dim,
    hidden_dim,
    progress=gr.Progress(),
):
    word2idx, idx2word, sequences = create_sequences(corpus)
    dataset = TextDataset(sequences, word2idx)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    vocab_size = len(word2idx)
    model = RNNModel(vocab_size, embedding_dim, hidden_dim)

    # Training the model
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    wandb.watch(model, log_freq=100)

    progress(0, desc="Starting training...", total=num_epochs)
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in progress.tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            hidden = model.init_hidden(batch_size=inputs.size(0))
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = loss_function(outputs, targets)
            wandb.log({"loss": loss.item()})
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        losses.append(average_loss)
        wandb.log({"epoch": epoch, "avg_loss": average_loss})
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
        progress(epoch + 1)

    progress(None)

    # Plotting the loss graph
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/loss_plot.png")
    plt.close()

    return model, word2idx, idx2word, "img/loss_plot.png"


# Function to generate text
def generate_text(seed_text, next_words, model, word2idx, idx2word):
    model.eval()
    words = seed_text.split()
    hidden = model.init_hidden(batch_size=1)
    for _ in range(next_words):
        input_seq = torch.tensor(
            [
                [
                    word2idx.get(word, word2idx["<UNK>"])
                    for word in words[-sequence_length:]
                ]
            ],
            dtype=torch.long,
        )
        output, hidden = model(input_seq, hidden)
        predicted_word_idx = output.argmax(dim=1).item()
        predicted_word = idx2word[predicted_word_idx]
        words.append(predicted_word)
    return " ".join(words)


# Gradio function to handle input and generate text
def gradio_fn(
    file,
    seed_text,
    next_words,
    num_epochs_input,
    embedding_dim_input,
    hidden_dim_input,
    learning_rate_input,
):
    global num_epochs, embedding_dim, hidden_dim, learning_rate, is_training_done, model, word2idx, idx2word, loss_plot

    num_epochs = int(num_epochs_input)
    embedding_dim = int(embedding_dim_input)
    hidden_dim = int(hidden_dim_input)
    learning_rate = float(learning_rate_input)

    corpus = handle_file_upload(file)
    if not is_training_done:
        model, word2idx, idx2word, loss_plot = train_model(
            corpus, num_epochs, learning_rate, embedding_dim, hidden_dim
        )
        is_training_done = True
    generated_text = generate_text(seed_text, next_words, model, word2idx, idx2word)

    return generated_text, loss_plot


# Function to reset training flag
def reset_training():
    global is_training_done
    is_training_done = False
    return "Training flag reset. Upload a new file to train the model."


# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_fn,
    inputs=[
        gr.File(label="Upload Text File"),
        gr.Textbox(lines=1, placeholder="Enter starting text", label="Starting Text"),
        gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            label="Number of Words to Generate",
            value=10,
        ),
        gr.Slider(minimum=1, maximum=200, step=1, label="Number of Epochs", value=30),
        gr.Slider(minimum=1, maximum=50, step=1, label="Embedding Dim", value=10),
        gr.Slider(minimum=1, maximum=200, step=1, label="Hidden Dim", value=50),
        gr.Number(label="Learning Rate", value=0.001),
    ],
    outputs=[gr.Textbox(label="Generated Text"), gr.Image(label="Training Loss")],
    title="RNN Text Generation with PyTorch",
    description="Upload a text file, input a starting text and the number of words to generate. The app uses a PyTorch RNN model to generate the next sequence of words.",
    theme="gradio/monochrome",
)

# Add a clear button to reset the training flag
# iface.clear_fn(reset_training)

# Launch the app
iface.queue().launch()

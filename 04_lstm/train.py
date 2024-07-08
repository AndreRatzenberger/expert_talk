import argparse
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

src = ""
name = ""
num_epochs = 10000
patience = 10  # Number of epochs to wait for improvement before stopping


class MusicDataset(Dataset):
    def __init__(self, network_input, network_output):
        self.network_input = network_input
        self.network_output = network_output

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], self.network_output[idx]


class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def train_network():
    notes = get_notes()
    n_vocab = len(set(notes))

    network_input, network_output, int_to_note = prepare_sequences(notes, n_vocab)
    train_size = int(0.8 * len(network_input))
    val_size = len(network_input) - train_size

    train_input, val_input = torch.utils.data.random_split(
        network_input, [train_size, val_size]
    )
    train_output, val_output = torch.utils.data.random_split(
        network_output, [train_size, val_size]
    )

    model = create_network(n_vocab)

    train(
        model, train_input, train_output, val_input, val_output, num_epochs, int_to_note
    )


def get_notes():
    notes = []
    files = glob.glob(src + "/*.mid")

    for file in tqdm(files, desc="Parsing MIDI files"):
        midi = converter.parse(file)
        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    with open("data/notes_" + name, "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    network_output = np.array(network_output)
    network_output = torch.tensor(network_output, dtype=torch.long)

    return torch.tensor(network_input, dtype=torch.float32), network_output, int_to_note


def create_network(n_vocab):
    model = MusicLSTM(input_size=1, hidden_size=512, output_size=n_vocab, num_layers=3)
    return model


def generate_music(model, network_input, int_to_note, sequence_length=100):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start].cpu().numpy()
    prediction_output = []

    model.eval()
    with torch.no_grad():
        for note_index in range(sequence_length):
            prediction_input = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                prediction_input = prediction_input.cuda()

            prediction = model(prediction_input)
            index = torch.argmax(prediction, dim=1).item()
            result = int_to_note[index]
            prediction_output.append(result)

            pattern = np.append(pattern[1:], index / float(len(int_to_note)))
            pattern = np.reshape(pattern, (pattern.shape[0], 1))

    return prediction_output


def create_midi(prediction_output, filename="output.mid"):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=filename)


def train(
    model,
    train_input,
    train_output,
    val_input,
    val_output,
    num_epochs,
    int_to_note,
    generation_interval=100,
):
    train_dataset = MusicDataset(train_input, train_output)
    val_dataset = MusicDataset(val_input, val_output)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    if torch.cuda.is_available():
        model = model.cuda()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch"
        ) as pbar:
            for inputs, labels in train_loader:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
                pbar.update()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(
                total=len(val_loader),
                desc=f"Validation {epoch}/{num_epochs}",
                unit="batch",
            ) as pbar:
                for inputs, labels in val_loader:
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    pbar.set_postfix({"val_loss": val_loss / (pbar.n + 1)})
                    pbar.update()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, epoch, avg_train_loss)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % generation_interval == 0:
            prediction_output = generate_music(model, train_input, int_to_note)
            create_midi(prediction_output, filename=f"output_epoch_{epoch}.mid")
            print(f"MIDI file generated for epoch {epoch}")

        if epoch % generation_interval == 0:
            sample_output = generate_music(
                model, train_input, int_to_note, sequence_length=20
            )
            print("Sample generated sequence:", sample_output[:20])


def save_checkpoint(model, epoch, loss):
    if not os.path.exists("weights_" + name):
        os.makedirs("weights_" + name)
    filepath = f"weights_{name}/weights-epoch-{epoch:04d}-loss-{loss:.4f}.pth"
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Music Generator")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing input music samples. eg: ./data",
    )
    parser.add_argument("--name", type=str, required=True, help="Name of the Music")
    parser.add_argument(
        "--epoch", type=int, default=10000, help="Number of training epochs"
    )
    args = parser.parse_args()
    src = args.input
    name = args.name
    num_epochs = args.epoch
    train_network()

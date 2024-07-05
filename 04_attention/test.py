import numpy as np


def get_random_embedding(dim=8):
    return np.random.rand(dim)


def attention(query, key, value):
    scores = np.dot(query, key.T)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.dot(weights, value)
    return output, weights


def feed_forward(x):
    W1 = np.random.rand(x.shape[-1], 16)  # First layer weights
    b1 = np.random.rand(16)  # First layer bias
    W2 = np.random.rand(16, x.shape[-1])  # Second layer weights
    b2 = np.random.rand(x.shape[-1])  # Second layer bias

    hidden = np.dot(x, W1) + b1
    hidden = np.maximum(0, hidden)  # ReLU activation
    output = np.dot(hidden, W2) + b2
    return output


input_sentence = "I love machine learning"
tokens = input_sentence.split()
print("Tokens:", tokens)

embeddings = [get_random_embedding() for _ in tokens]
print("Embeddings:", embeddings)

query = np.random.rand(8)  # Random query vector for simplicity
keys = np.array(embeddings)
values = np.array(embeddings)

attention_output, attention_weights = attention(query, keys, values)
print("Attention Output:", attention_output)
print("Attention Weights:", attention_weights)

transformer_output = feed_forward(attention_output)
print("Transformer Output:", transformer_output)

output_tokens = ["J'", "aime", "l'apprentissage", "automatique"]
output_sentence = " ".join(output_tokens)
print("Translated Sentence:", output_sentence)

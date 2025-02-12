import torch
import numpy as np
import torch.nn as nn

# Indeksering av karakterer og generering av one-hot encoding
index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']
char_encodings = np.eye(len(index_to_char))  #lager matrise med one-hot encoding
encoding_size = len(char_encodings)  #størrelse på encoding lik antall karakterer

# Treningsdata for sekvensen ' hello world '
x_train = torch.tensor([
    [char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]],
    [char_encodings[4]], [char_encodings[0]], [char_encodings[5]], [char_encodings[4]], [char_encodings[6]],
    [char_encodings[3]], [char_encodings[7]]], dtype=torch.float)  # Input: ' hello world'

y_train = torch.tensor([
    char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4],
    char_encodings[0], char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3],
    char_encodings[7], char_encodings[0]], dtype=torch.float)  # Output: 'hello world '


# LSTM-modell
class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  # LSTM-lag med 128 skjulte noder
        self.dense = nn.Linear(128, encoding_size)  # Fullt tilkoblet lag som mappes til encoding_size

    def reset(self):  # Nullstiller skjult tilstand før ny sekvens
        self.hidden_state = torch.zeros(1, 1, 128)  # Skjult tilstand
        self.cell_state = torch.zeros(1, 1, 128)  # Cell tilstand

    def logits(self, x):  # Beregner logits
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # Softmax for å beregne sannsynligheter
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # Beregner kryssentropitapet
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


# Opprett modell
model = LongShortTermMemoryModel(encoding_size)

# Optimalisering med RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)

# Treningssløyfe
for epoch in range(500):
    model.reset()  # Nullstill tilstander før hver epoke
    model.loss(x_train, y_train).backward()  # Beregn tap og utfør backpropagation
    optimizer.step()  # Oppdater vektene
    optimizer.zero_grad()  # Nullstill gradientene

    # Generer tekst etter hver 10. epoke
    if epoch % 10 == 9:
        model.reset()
        text = ' h'
        model.f(torch.tensor([[char_encodings[0]]], dtype=torch.float))
        y = model.f(torch.tensor([[char_encodings[1]]], dtype=torch.float))
        text += index_to_char[y.argmax(1)]

        # Generer 50 bokstaver
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]], dtype=torch.float))
            text += index_to_char[y.argmax(1)]

        print(f'Epoke {epoch + 1}, Generert tekst: {text}')  # Print tekst for hver 10. epoke


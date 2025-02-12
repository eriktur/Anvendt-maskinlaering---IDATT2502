import torch
import torch.nn as nn


char_encodings = {
    ' ': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'a': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'c': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'f': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'h': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    'l': [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    'm': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    'n': [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    'o': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    'p': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    'r': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    's': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    't': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
}
encoding_size = len(next(iter(char_encodings.values())))

# Definer emojis
emojis = ['🎩', '🐀', '🐱', '🏠', '👨‍🦱', '🧢', '👦']
emoji_to_idx = {emoji: i for i, emoji in enumerate(emojis)}

words = ['hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son '] # Ordene vi skal klassifisere
y_train = torch.tensor([emoji_to_idx['🎩'], emoji_to_idx['🐀'], emoji_to_idx['🐱'], emoji_to_idx['🏠'],
                        emoji_to_idx['👨‍🦱'], emoji_to_idx['🧢'], emoji_to_idx['👦']])

# Konverter inputord til sekvens av bokstavkodinger til bruk i LSTM
x_train = torch.tensor([[char_encodings[char] for char in word] for word in words])
x_train = x_train.permute(1, 0, 2)  # Reorganiser dimensjoner til (sequence length, batch size, encoding size)

# Definer modellen (many-to-one LSTM)
class ManyToOneLSTM(nn.Module): # Mange-til-en LSTM
    def __init__(self, encoding_size, hidden_size, output_size):
        super(ManyToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(encoding_size, hidden_size) # LSTM-lag
        self.dense = nn.Linear(hidden_size, output_size)  # Fullt tilkoblet lag

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # Fokuser på siste skjulte tilstand
        return self.dense(hn.squeeze(0))  # Bruk den siste skjulte tilstanden for å klassifisere

# Opprett LSTM-modellen
model = ManyToOneLSTM(encoding_size, 128, len(emojis))

# Treningsoppsett
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Trening av modellen
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoke {epoch}, Tap: {loss.item()}')

# Teste modellen på nye ord
def predict_emoji(word):
    x_test = torch.tensor([[char_encodings[char] for char in word]])
    x_test = x_test.permute(1, 0, 2)  # Reorganiser dimensjoner til (sequence length, batch size, encoding size)
    with torch.no_grad():
        output = model(x_test)
        emoji_idx = output.argmax().item()
    return emojis[emoji_idx]

print(f'Prediksjon for "rt  ": {predict_emoji("at ")}')
print(f'Prediksjon for "rats": {predict_emoji("rats")}')

import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

# --- 1. Carregar os feedbacks ---
df = pd.read_csv("feedback.csv")

# --- 2. Pré-processamento ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Texto_Feedback"]).toarray()

labels = {"Negativo": 0, "Neutro": 1, "Positivo": 2}
y = df["Sentimento_Real"].map(labels).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Modelo simples ---
class SentimentNet(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc2(self.relu(self.fc1(x))))

model = SentimentNet(input_size=X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 4. Treinar rede simples ---
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

print("✅ Modelo treinado!")

# --- 5. Salvar modelo e vetorizer ---
torch.save(model.state_dict(), "sentimento_model.pt")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Modelo e vectorizer salvos para Flask")

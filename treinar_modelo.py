import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Rede Neural ---
class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Haversine ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- Carrega destinos ---
def carregar_destinos_robustos():
    df = pd.read_csv("destinos_turisticos.csv")
    return df

# --- Treina modelo ---
def treinar_modelo(destinos_df):
    cat2idx = {c:i for i,c in enumerate(destinos_df["Tipo_Ecosistema"].unique())}
    prov2idx = {p:i for i,p in enumerate(destinos_df["Provincia"].unique())}
    
    X = []
    y = []
    max_dist = destinos_df["Distancia_km"].max()
    max_sim = destinos_df["Similaridade"].max()
    
    for _, row in destinos_df.iterrows():
        cat_idx = cat2idx[row["Tipo_Ecosistema"]]
        prov_idx = prov2idx[row["Provincia"]]
        distancia_norm = row["Distancia_km"] / max_dist
        sim_norm = row["Similaridade"] / max_sim
        X.append([cat_idx, prov_idx, distancia_norm, sim_norm])
        y.append(row.get("Pontuacao", 1.0))
    
    X = np.array(X)
    y = np.array(y)
    
    # --- Divisão treino/teste ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    model = Regressor(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # --- Avaliação ---
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))   # compatível com todas as versões
    r2 = r2_score(y_test, y_pred)
    
    print("\n=== Avaliação do Modelo ===")
    print(f"MAE  (Erro Médio Absoluto): {mae:.4f}")
    print(f"RMSE (Erro Quadrático Médio): {rmse:.4f}")
    print(f"R²   (Coeficiente de Determinação): {r2:.4f}")
    print(f"Precisão aproximada: {r2 * 100:.2f}%")
    
    # --- Gráfico Real vs Previsto ---
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())],
             [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())],
             color='red', linestyle='--')
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Comparação: Real vs Previsto")
    plt.tight_layout()
    plt.show()
    
    return model, cat2idx, prov2idx

# --- Segunda opinião ---
def segunda_opiniao(model, cat2idx, prov2idx, destinos_df, categoria, latitude, longitude):
    max_dist = destinos_df["Distancia_km"].max()
    X_test = []
    for _, row in destinos_df.iterrows():
        cat_idx = cat2idx.get(row["Tipo_Ecosistema"], 0)
        prov_idx = prov2idx.get(row["Provincia"], 0)
        dist_km = haversine(latitude, longitude, row["Latitude"], row["Longitude"])
        distancia_norm = dist_km / max_dist
        sim_norm = row["Similaridade"] / (destinos_df["Similaridade"].max() if destinos_df["Similaridade"].max()>0 else 1.0)
        X_test.append([cat_idx, prov_idx, distancia_norm, sim_norm])
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_test).numpy().flatten()
    
    destinos_df = destinos_df.copy()
    destinos_df["Pontuacao_Final"] = pred
    
    destinos_filtrados = destinos_df[destinos_df["Tags"].str.contains(categoria, case=False, na=False)]
    if destinos_filtrados.empty:
        destinos_filtrados = destinos_df
    
    top3 = destinos_filtrados.sort_values("Pontuacao_Final", ascending=False).head(3)
    return top3[["Nome", "Provincia", "Tipo_Ecosistema", "Pontuacao_Final"]]

# --- EXECUÇÃO ---
if __name__ == "__main__":
    destinos_df = carregar_destinos_robustos()
    user_lat, user_lon = -8.8383, 13.2345
    
    destinos_df["Distancia_km"] = destinos_df.apply(
        lambda row: haversine(user_lat, user_lon, row["Latitude"], row["Longitude"]), axis=1
    )
    tfidf = TfidfVectorizer()
    matriz_tfidf = tfidf.fit_transform(destinos_df["Tags"])
    consulta = tfidf.transform(destinos_df["Tags"])
    destinos_df["Similaridade"] = cosine_similarity(consulta, matriz_tfidf).diagonal()
    
    destinos_df["Pontuacao"] = 0.6*destinos_df["Similaridade"] + 0.4*(1 - destinos_df["Distancia_km"]/destinos_df["Distancia_km"].max())
    
    model, cat2idx, prov2idx = treinar_modelo(destinos_df)
    
    top3 = segunda_opiniao(model, cat2idx, prov2idx, destinos_df, categoria="aventura",
                           latitude=user_lat, longitude=user_lon)
    print("\n=== Top 3 Recomendações ===")
    print(top3)

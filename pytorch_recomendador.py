import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Rede simples para regressão de pontuação
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

# Treina o modelo usando os destinos filtrados
def treinar_modelo(destinos_df):
    # Map tipo ecossistema e provincia para números
    cat2idx = {c:i for i,c in enumerate(destinos_df["Tipo_Ecosistema"].unique())}
    prov2idx = {p:i for i,p in enumerate(destinos_df["Provincia"].unique())}
    
    X = []
    y = []
    
    # Normaliza distância e similaridade
    max_dist = destinos_df["Distancia_km"].max() if "Distancia_km" in destinos_df else 1.0
    max_sim = destinos_df["Similaridade"].max() if "Similaridade" in destinos_df else 1.0
    
    for _, row in destinos_df.iterrows():
        cat_idx = cat2idx[row["Tipo_Ecosistema"]]
        prov_idx = prov2idx[row["Provincia"]]
        distancia_norm = row.get("Distancia_km", 1.0)/max_dist
        sim_norm = row.get("Similaridade", 1.0)/max_sim
        X.append([cat_idx, prov_idx, distancia_norm, sim_norm])
        y.append(row.get("Pontuacao", 1.0))
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1,1)
    
    model = Regressor(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Treinamento simples
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    return model, cat2idx, prov2idx

# Função para prever a segunda opinião
def segunda_opiniao(model, cat2idx, prov2idx, destinos_df, categoria, latitude, longitude):
    # Calcula distância normalizada
    max_dist = destinos_df["Distancia_km"].max()
    X_test = []
    for _, row in destinos_df.iterrows():
        cat_idx = cat2idx.get(row["Tipo_Ecosistema"], 0)
        prov_idx = prov2idx.get(row["Provincia"], 0)
        dist_km = haversine(latitude, longitude, row["Latitude"], row["Longitude"])
        distancia_norm = dist_km / max_dist
        sim_norm = row.get("Similaridade", 1.0) / (destinos_df["Similaridade"].max() if destinos_df["Similaridade"].max()>0 else 1.0)
        X_test.append([cat_idx, prov_idx, distancia_norm, sim_norm])
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_test).numpy().flatten()
    
    destinos_df = destinos_df.copy()
    destinos_df["Pontuacao_Final"] = pred
    
    # Filtra pela categoria (tags)
    destinos_filtrados = destinos_df[destinos_df["Tags"].str.contains(categoria, case=False, na=False)]
    if destinos_filtrados.empty:
        destinos_filtrados = destinos_df
    
    top3 = destinos_filtrados.sort_values("Pontuacao_Final", ascending=False).head(3)
    
    return top3[["Nome", "Provincia", "Tipo_Ecosistema", "Pontuacao_Final"]]
    

# Função Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

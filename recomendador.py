import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def recomendar_destinos(preferencias, destinos_df, user_lat, user_lon, max_impacto=3.0, top_n=5):
    destinos_filtrados = destinos_df[destinos_df["Impacto_Ambiental"] <= max_impacto].copy()

    # Similaridade de tags
    tfidf = TfidfVectorizer()
    matriz_tfidf = tfidf.fit_transform(destinos_filtrados["Tags"])
    consulta = tfidf.transform([preferencias])
    destinos_filtrados["Similaridade"] = cosine_similarity(consulta, matriz_tfidf).flatten()

    # Distância geográfica
    destinos_filtrados["Distancia_km"] = destinos_filtrados.apply(
        lambda row: haversine(user_lat, user_lon, row["Latitude"], row["Longitude"]), axis=1
    )

    # Combina similaridade e proximidade
    destinos_filtrados["Distancia_norm"] = 1 - (destinos_filtrados["Distancia_km"] / destinos_filtrados["Distancia_km"].max())
    destinos_filtrados["Pontuacao"] = 0.6*destinos_filtrados["Similaridade"] + 0.4*destinos_filtrados["Distancia_norm"]

    recomendacoes = destinos_filtrados.sort_values(by="Pontuacao", ascending=False).head(top_n)
    return recomendacoes[[
        "Nome","Provincia","Tipo_Ecosistema","Custo_Medio","Impacto_Ambiental",
        "Visitantes_Anuais","Tags","Distancia_km","Similaridade","Pontuacao"
    ]]

def carregar_destinos_robustos():
    return pd.DataFrame([
        {"ID_destino": 1, "Nome": "Parque Kissama", "Provincia": "Luanda", "Tipo_Ecosistema": "savanna", 
         "Custo_Medio": 50, "Impacto_Ambiental": 2.5, "Latitude": -9.4389, "Longitude": 13.0763, 
         "Visitantes_Anuais": 50000, "Tags": "natureza, safari, ecoturismo"},
        {"ID_destino": 2, "Nome": "Miradouro da Lua", "Provincia": "Luanda", "Tipo_Ecosistema": "deserto", 
         "Custo_Medio": 10, "Impacto_Ambiental": 1.0, "Latitude": -9.4012, "Longitude": 13.1652, 
         "Visitantes_Anuais": 40000, "Tags": "paisagem, fotografia, cultural"},
        {"ID_destino": 3, "Nome": "Reserva de Cangandala", "Provincia": "Malanje", "Tipo_Ecosistema": "floresta", 
         "Custo_Medio": 75, "Impacto_Ambiental": 4.0, "Latitude": -9.8001, "Longitude": 16.7004, 
         "Visitantes_Anuais": 25000, "Tags": "fauna, floresta, ecoturismo"},
        {"ID_destino": 4, "Nome": "Praia do Mussulo", "Provincia": "Luanda", "Tipo_Ecosistema": "costeiro", 
         "Custo_Medio": 40, "Impacto_Ambiental": 1.5, "Latitude": -8.9767, "Longitude": 13.0509, 
         "Visitantes_Anuais": 60000, "Tags": "praia, relaxamento, aquático"},
        {"ID_destino": 5, "Nome": "Centro Histórico de Mbanza Kongo", "Provincia": "Zaire", "Tipo_Ecosistema": "urbano", 
         "Custo_Medio": 25, "Impacto_Ambiental": 0.5, "Latitude": -6.2664, "Longitude": 14.24, 
         "Visitantes_Anuais": 30000, "Tags": "património, história, cultural"},
        {"ID_destino": 6, "Nome": "Parque Nacional da Quiçama", "Provincia": "Bengo", "Tipo_Ecosistema": "savanna", 
         "Custo_Medio": 60, "Impacto_Ambiental": 3.0, "Latitude": -9.5600, "Longitude": 13.0000, 
         "Visitantes_Anuais": 20000, "Tags": "natureza, safari, fauna"},
        {"ID_destino": 7, "Nome": "Praia de Cabo Ledo", "Provincia": "Bengo", "Tipo_Ecosistema": "costeiro", 
         "Custo_Medio": 30, "Impacto_Ambiental": 1.2, "Latitude": -9.5000, "Longitude": 13.5000, 
         "Visitantes_Anuais": 45000, "Tags": "praia, surf, relaxamento"},
        {"ID_destino": 8, "Nome": "Fortaleza de São Miguel", "Provincia": "Luanda", "Tipo_Ecosistema": "urbano", 
         "Custo_Medio": 15, "Impacto_Ambiental": 0.8, "Latitude": -8.8383, "Longitude": 13.2345, 
         "Visitantes_Anuais": 35000, "Tags": "história, cultural, arquitetura"},
        {"ID_destino": 9, "Nome": "Serra da Leba", "Provincia": "Huíla", "Tipo_Ecosistema": "montanha", 
         "Custo_Medio": 35, "Impacto_Ambiental": 2.0, "Latitude": -14.917, "Longitude": 13.983, 
         "Visitantes_Anuais": 40000, "Tags": "paisagem, aventura, trilha"},
        {"ID_destino": 10, "Nome": "Parque Nacional da Kissama II", "Provincia": "Luanda", "Tipo_Ecosistema": "savanna", 
         "Custo_Medio": 55, "Impacto_Ambiental": 3.2, "Latitude": -9.4200, "Longitude": 13.0900, 
         "Visitantes_Anuais": 22000, "Tags": "safari, fauna, natureza"},
        {"ID_destino": 11, "Nome": "Ponto Turístico Calandula", "Provincia": "Malanje", "Tipo_Ecosistema": "cachoeira", 
         "Custo_Medio": 20, "Impacto_Ambiental": 1.5, "Latitude": -9.550, "Longitude": 16.350, 
         "Visitantes_Anuais": 30000, "Tags": "paisagem, fotografia, aventura"},
        {"ID_destino": 12, "Nome": "Ilha do Mussulo", "Provincia": "Luanda", "Tipo_Ecosistema": "costeiro", 
         "Custo_Medio": 45, "Impacto_Ambiental": 1.8, "Latitude": -9.000, "Longitude": 13.100, 
         "Visitantes_Anuais": 50000, "Tags": "praia, relaxamento, ecoturismo"},
        {"ID_destino": 13, "Nome": "Parque Nacional da Cameia", "Provincia": "Moxico", "Tipo_Ecosistema": "floresta", 
         "Custo_Medio": 70, "Impacto_Ambiental": 3.5, "Latitude": -14.0, "Longitude": 21.0, 
         "Visitantes_Anuais": 18000, "Tags": "natureza, fauna, ecoturismo"},
        {"ID_destino": 14, "Nome": "Caverna Tundavala", "Provincia": "Huíla", "Tipo_Ecosistema": "montanha", 
         "Custo_Medio": 25, "Impacto_Ambiental": 1.5, "Latitude": -14.95, "Longitude": 13.98, 
         "Visitantes_Anuais": 15000, "Tags": "aventura, trilha, paisagem"}
    ])

import pandas as pd

# --- 1. Base de Destinos Turísticos ---
data_destino = {
    'ID_destino': [1, 2, 3, 4, 5],
    'Nome': [
        'Parque Kissama',
        'Miradouro da Lua',
        'Reserva de Cangandala',
        'Praia do Mussulo',
        'Centro Histórico de Mbanza Kongo'
    ],
    'Provincia': ['Luanda', 'Luanda', 'Malanje', 'Luanda', 'Zaire'],
    'Tipo_Ecosistema': ['savanna', 'deserto', 'floresta', 'costeiro', 'urbano'],
    'Custo_Medio': [50, 10, 75, 40, 25],
    'Impacto_Ambiental': [2.5, 1.0, 4.0, 1.5, 0.5],
    'Latitude': [-9.4389, -9.4012, -9.8001, -8.9767, -6.2664],
    'Longitude': [13.0763, 13.1652, 16.7004, 13.0509, 14.2400],
    'Visitantes_Anuais': [50000, 40000, 25000, 60000, 30000],
    'Tags': [
        'natureza, safari, ecoturismo',
        'paisagem, fotografia, cultural',
        'fauna, floresta, ecoturismo',
        'praia, relaxamento, aquático',
        'património, história, cultural'
    ]
}
df_destino = pd.DataFrame(data_destino)
df_destino.to_csv("destinos.csv", index=False)
print("✅ Ficheiro 'destinos.csv' criado.")


# --- 2. Base de Perfis de Turistas ---
data_turista = {
    'ID_turista': [101, 102, 103],
    'Nome': ['Jorge', 'Maria', 'Pedro'],
    'Preferencias': [
        'ecoturismo, natureza, safari',
        'cultural, história, património',
        'praia, relaxamento, aquático'
    ]
}
df_turista = pd.DataFrame(data_turista)
df_turista.to_csv("perfil_turista.csv", index=False)
print("✅ Ficheiro 'perfil_turista.csv' criado.")


# --- 3. Base de Feedbacks ---
data_feedback = {
    'ID_feedback': [1, 2, 3, 4, 5],
    'Destino': [
        'Parque Kissama',
        'Miradouro da Lua',
        'Reserva de Cangandala',
        'Praia do Mussulo',
        'Centro Histórico de Mbanza Kongo'
    ],
    'Texto_Feedback': [
        "Excelente, natureza bem preservada, vou voltar!",
        "O local é lindo, mas a infraestrutura deixa a desejar.",
        "Experiência terrível, muita poluição e pouco apoio local.",
        "Um ponto turístico imperdível, recomendo a todos!",
        "Não foi mau, mas esperava mais pelo preço cobrado."
    ],
    'Sentimento_Real': ['Positivo', 'Neutro', 'Negativo', 'Positivo', 'Neutro']
}
df_feedback = pd.DataFrame(data_feedback)
df_feedback.to_csv("feedback.csv", index=False)
print("✅ Ficheiro 'feedback.csv' criado.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar a base de dados
# Certifique-se de que o arquivo 'iris.csv' está na mesma pasta do seu script
try:
    df = pd.read_csv('iris.csv')
    print("Base de dados Iris carregada com sucesso!")
    print(df.head()) # Mostra as primeiras 5 linhas do DataFrame
except FileNotFoundError:
    print("Erro: Arquivo 'iris.csv' não encontrado. Certifique-se de que está na mesma pasta.")
    exit()

# 2. Pré-processar os dados: Separar features (X) e target (y)
# ATENÇÃO: Os nomes das colunas devem corresponder EXATAMENTE aos nomes do seu CSV.
# Para o CSV que você usou, as colunas são 'sepal.length', 'sepal.width', etc.
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']] # Features
y = df['variety'] # Target (o nome da coluna target no seu CSV é 'variety')

# Converter as classes de texto (Setosa, Versicolor, Virginica) para números
# Isso é necessário para muitos algoritmos de Machine Learning
le = LabelEncoder()
y = le.fit_transform(y)
# Agora y conterá 0, 1 ou 2 em vez dos nomes das espécies

print("\nFormato dos dados (X, y):")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes numéricas: {le.classes_}") # Mostra o mapeamento de números para nomes

# 3. Separar a base de dados em treino e teste
# Usamos 80% dos dados para treino e 20% para teste
# random_state garante que a separação seja a mesma toda vez que você rodar o código
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDados separados em treino e teste:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 4. Criar e treinar o modelo
# Instanciamos o modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42)

# Treinamos o modelo com os dados de treino (X_train, y_train)
model.fit(X_train, y_train)

print("\nModelo de Árvore de Decisão treinado com sucesso!")

# 5. Fazer previsões nos dados de teste
y_pred = model.predict(X_test)

# 6. Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAcurácia do modelo na base de testes: {accuracy:.2f}")

# Exemplo de previsão para algumas amostras de teste
print("\nPrevisões vs. Valores Reais para algumas amostras de teste:")
# Mapeia os números de volta para os nomes das espécies para melhor legibilidade na saída
predicted_species_names = le.inverse_transform(y_pred)
actual_species_names = le.inverse_transform(y_test)

for i in range(5): # Mostra as primeiras 5 previsões
    print(f"Amostra {i+1}: Previsto: {predicted_species_names[i]}, Real: {actual_species_names[i]}")
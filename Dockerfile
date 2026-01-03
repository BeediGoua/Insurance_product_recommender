# Utiliser une image Python officielle légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande de lancement
CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. On crée l'utilisateur en premier
RUN useradd -m appuser
WORKDIR /app

# 2. On copie TOUT le code (requis pour que pip lise le pyproject.toml)
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser backend.py .
COPY --chown=appuser:appuser data/ ./data/

# 3. 🚀 LE BYPASS DU PARE-FEU : On utilise pip classique avec les exceptions SSL
# 'pip install .' va lire le pyproject.toml et installer ton projet et ses dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    .

# 4. On bascule sur l'utilisateur sécurisé
USER appuser

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
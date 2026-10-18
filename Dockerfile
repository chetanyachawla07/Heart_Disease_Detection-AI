# Serves the Flask app with gunicorn:   docker build -t heart-app . && docker run -p 5000:5000 heart-app
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Keep scikit-learn at the version the committed model was trained with
# (the app warns on a mismatch; retrain with `python train.py` if you change it).
RUN pip install --no-cache-dir flask gunicorn pandas numpy joblib "scikit-learn==1.8.0"

COPY heart_disease ./heart_disease
COPY webapp ./webapp
COPY models ./models

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "webapp.app:create_app()"]

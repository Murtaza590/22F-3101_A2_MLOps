FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app.py model.tflite model_info.json ./
USER user
ENV PORT=10000
EXPOSE 10000
CMD exec gunicorn -b 0.0.0.0:${PORT} -w 1 --timeout 120 app:app
FROM python:3.10
WORKDIR /app

COPY src/predict_app.py ./src/predict_app.py
COPY models/catboost_v1.joblib ./models/catboost_v1.joblib
COPY req_docker.txt ./
COPY .env ./

RUN python3 -m pip install --upgrade pip
RUN pip install -r req_docker.txt

CMD ["gunicorn", "-b", "0.0.0.0", "src.predict_app:app", "--capture-output"]
EXPOSE 8000
FROM python:3.11

RUN python -m pip install fastapi uvicorn pandas scikit-learn

WORKDIR /app

ADD fastapi_model.py fastapi_model.py
ADD HD_model.joblib HD_model.joblib

EXPOSE 8000

CMD ["uvicorn", "fastapi_model:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
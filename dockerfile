FROM python:3.9

WORKDIR /softwear_technology

COPY requirements.txt /softwear_technology/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /softwear_technology/

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "main.py"]

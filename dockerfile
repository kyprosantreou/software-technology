FROM python:3.9

WORKDIR /software-technology

COPY requirements.txt /software-technology/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /software-technology/

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "main.py"]
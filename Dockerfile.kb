# 
FROM python:3.9.9

# 
WORKDIR /code

# sudo apt-get update.
# sudo apt-get upgrade.
# sudo apt install poppler-utils.

# 
COPY requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install --no-cache-dir --upgrade chromadb bs4 unstructured[local-inference]

# 
COPY ./app /code/app

ENV PYTHONPATH "${PYTHONPATH}:/code/app"


# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
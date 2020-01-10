FROM python:3.7

RUN pip install pipenv
# RUN apt-get update && apt-get install -y libhdf5-dev

WORKDIR /app

COPY Pipfile /app/
RUN pipenv install --pre
RUN pipenv install --system --deploy

COPY . /app/
CMD /app/simulate.py

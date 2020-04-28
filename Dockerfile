FROM python:alpine3.7
RUN pip install -r requirements.txt
CMD ./run.sh

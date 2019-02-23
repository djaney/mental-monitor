FROM petronetto/opencv-alpine
ENV FLASK_APP=faces.py
ENV FLASK_ENV=production

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD flask run --host 0.0.0.0 --port 9901 --no-reload
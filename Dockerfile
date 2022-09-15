#v2
FROM python:3.8
COPY main.py ./main.py
COPY /data/Gdansk.h5 /data/Gdansk.h5
COPY /utils /utils
COPY requirements.txt .
RUN pip install -Uqr requirements.txt
CMD ["python", "main.py"]


# ------ Build new image and tag it --------------
# docker build --no-cache -t ais:v2 .
# ------ Run container from that image -----------
# docker container run -v $(pwd)/output:/output --name ais ais:v2
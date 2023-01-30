FROM python:latest
WORKDIR /mydata
COPY . .
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
COPY hldmdan.py ./hldmdan.py
CMD python /mydata/hldmdan.py
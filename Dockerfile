FROM python:3.8-slim-buster
WORKDIR /app
COPY . .
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt install gcc -y
RUN apt-get install python3-dev -y
RUN export PYTHONPATH=$PYTHONPATH:/app
RUN pip3 install -r requirements.txt
# RUN python ./recsys/run_compile_all_cython.py
CMD ["/bin/bash"]
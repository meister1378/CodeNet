FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY infer/ infer/
COPY checkpoints/ checkpoints/

WORKDIR /app/infer
ENTRYPOINT ["python", "infer.py"]

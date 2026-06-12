FROM rocm/pytorch-nightly:20260429082157-rocm7.2.2

WORKDIR /workspace

COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
RUN apt update -y && apt install -y jq

COPY . /workspace/torchtitan
RUN pip install --no-cache-dir -e /workspace/torchtitan

CMD ["bash"]

FROM nvcr.io/nvidia/tritonserver:26.07-py3

WORKDIR /app
COPY . .

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
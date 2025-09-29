FROM python:3.11-slim

RUN apt update && apt install -y \
   tk tk-dev curl jq procs
RUN pip install --no-cache-dir \
   boto3 \
   dotenv \
   fastapi \
   requests \
   uvicorn \
   jaeger-client>=4.8.0 \
   opentracing>=2.4.0 \
   tk

COPY ./example_server.py /app/example_server.py
COPY ./default_query_constructor.py /app/default_query_constructor.py
COPY ./base_query_constructor.py /app/base_query_constructor.py
EXPOSE 8000

CMD ["python", "/app/example_server.py"]

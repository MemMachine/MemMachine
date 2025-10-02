FROM python:3.11-slim

RUN apt update && apt install -y \
   tk tk-dev curl jq procs
RUN pip install --no-cache-dir \
   boto3 \
   dotenv \
   fastapi \
   requests \
   tk \
   fastapi>=0.111 \
   uvicorn[standard]>=0.30 \
   jaeger-client>=4.8 \
   opentracing>=2.4 \
   prometheus-client>=0.20
COPY ./example_server.py /app/example_server.py
COPY ./default_query_constructor.py /app/default_query_constructor.py
COPY ./base_query_constructor.py /app/base_query_constructor.py
COPY observability.py /app/observability.py
EXPOSE 8000

CMD ["python", "/app/example_server.py"]

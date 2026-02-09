
import json

from fastapi import FastAPI
from memmachine.server.api_v2.router import load_v2_api_router

def generate_openapi():
    app = FastAPI(servers=[{"url": "https://localhost:8080"}])
    load_v2_api_router(app)

    openapi_schema = app.openapi()
    print(json.dumps(openapi_schema, indent=2))

if __name__ == "__main__":
    generate_openapi()


import json
import sys
import os

# Add src to sys.path so we can import memmachine
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from fastapi import FastAPI
from memmachine.server.api_v2.router import load_v2_api_router

def generate_openapi():
    app = FastAPI()
    load_v2_api_router(app)
    
    openapi_schema = app.openapi()
    print(json.dumps(openapi_schema, indent=2))

if __name__ == "__main__":
    generate_openapi()

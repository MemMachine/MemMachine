# Helm Deployment

```bash
helm upgrade --install memmachine ./helm \
    --create-namespace --namespace memmachine \
    --set backend.config.openai_api_key=sk-proj-YOURKEY
```
# Helm Deployment

To deploy the chart, we add the OpenAI api key on the fly.

```bash
helm upgrade --install memmachine ./helm \
    --create-namespace --namespace memmachine \
    --set backend.config.openai_api_key=${OPENAI_API_KEY} \
    --set 'open-webui.extraEnvVars[0].name=OPENAI_API_KEY' \
    --set "open-webui.extraEnvVars[0].value=${OPENAI_API_KEY}" \
    --set=chatbot.openaiApiKey=${OPENAI_API_KEY}
```

## Additional Services

### Jaeger Tracing

The middleware has a first attempt of using OpenTelemetry tracing. For that you can enable the deployment of a jaeger-all-in-one stack within the values.yaml

```yaml
jaeger:
  enabled: false
  name: jaeger
  service:
    type: ClusterIP
  ingress:
    enabled: true
    className: ""
    annotations: {}
      # kubernetes.io/ingress.class: nginx
      # cert-manager.io/cluster-issuer: letsencrypt
      # nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
      # nginx.ingress.kubernetes.io/from-to-www-redirect: "true"
    hosts:
      - host: jaeger.rack
        paths:
          - path: /
            pathType: ImplementationSpecific
    tls: []
  volume:
    enabled: true
    className: "nfs-client"
    size: 3Gi
```
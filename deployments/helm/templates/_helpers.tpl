{{/*
memmachine.ltmBackend
Returns episodicMemory.longTermMemory.backend, failing the render when it is
anything other than "declarative" or "event". Every template that branches on
the backend goes through this, so a typo stops the install instead of silently
falling through to one of the branches.
*/}}
{{- define "memmachine.ltmBackend" -}}
{{- $backend := .Values.episodicMemory.longTermMemory.backend | toString -}}
{{- if not (has $backend (list "declarative" "event")) -}}
{{- fail (printf "episodicMemory.longTermMemory.backend must be \"declarative\" or \"event\", got %q" $backend) -}}
{{- end -}}
{{- $backend -}}
{{- end -}}

{{/*
memmachine.qdrantApiKeySecret
Name of the Secret holding QDRANT_API_KEY, or "" when no API key is configured.
qdrant.existingSecret wins over qdrant.apiKey; the chart creates qdrant-secret
only for the latter.
*/}}
{{- define "memmachine.qdrantApiKeySecret" -}}
{{- if .Values.qdrant.existingSecret -}}
{{- .Values.qdrant.existingSecret -}}
{{- else if .Values.qdrant.apiKey -}}
qdrant-secret
{{- end -}}
{{- end -}}

{{/*
memmachine.appEnv and memmachine.appVolumeMounts
The environment and mounts of the server container and of the migrate init
container, so both read the same configuration.
*/}}
{{- define "memmachine.appEnv" -}}
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: memmachine-secrets
      key: OPENAI_API_KEY
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: POSTGRES_PASSWORD
- name: NEO4J_USER
  valueFrom:
    secretKeyRef:
      name: neo4j-secret
      key: NEO4J_USER
- name: NEO4J_PASSWORD
  valueFrom:
    secretKeyRef:
      name: neo4j-secret
      key: NEO4J_PASSWORD
{{- with include "memmachine.qdrantApiKeySecret" . }}
- name: QDRANT_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ . }}
      key: QDRANT_API_KEY
{{- end }}
- name: LOG_FILE
  value: "/app/data/memmachine.log"
- name: MEMORY_CONFIG
  value: {{ .Values.memmachine.config.memoryConfigPath }}
{{- end -}}

{{- define "memmachine.appVolumeMounts" -}}
- name: config-volume
  mountPath: /app/configuration.yml
  subPath: configuration.yml
- name: env-volume
  mountPath: /app/.env
  subPath: .env
- name: data-volume
  mountPath: /app/data
{{- end -}}

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

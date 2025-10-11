# Local Models

To run OpenAI API compatible models we have muliple options at hand.
- run vllm
- run litellm backed by Ollama
- ...

Since Ollama is a popular way of serving models locally, we'll use Ollama with a litellm proxy.

## Install

We'll use a python environment to encapsulte the install from other environments.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip3 install "litellm[proxy]"
```

## Ollama

Install [Ollama](https://docs.ollama.com/quickstart) on your computer.

Afterwards download an embedding and llm model.

```
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

## LiteLLM

Next, we are going to proxy both models with litellm.

```
cat > litellm.yaml << EOF
model_list:
  - model_name: qwen3:8b
    litellm_params:
      model: ollama/qwen3:8b
      api_base: http://localhost:11434
  - model_name: qwen3-embedding:0.6b
    litellm_params:
      model: ollama/qwen3-embedding:0.6b
      api_base: http://localhost:11434
litellm_settings:
    drop_params: true
EOF
litellm --config config.yaml --port 8000 --telemetry=False --detailed_debug --debug
```

## configuration.yml

Change the config blogs for the `embedder` and the `testmodel`.
```
Model:
  testmodel:
    model_vendor: openai-compatible
    model_name: "qwen3:8b"
    model: "qwen3:8b"
    api_key: "EMPTY"
    base_url: http://host.docker.internal:8000

embedder:
  my_embedder_id:
    model_vendor: openai-compatible
    model_name: "qwen3-embedding:0.6b"
    api_key: "EMPTY"
    base_url: "http://host.docker.internal:8000"
```
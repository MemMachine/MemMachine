import json
import os
import time

import boto3
import openai
from dotenv import load_dotenv
from model_config import MODEL_TO_INFERENCE_PROFILE_ARN, MODEL_TO_PROVIDER

# Lazy initialization of Google Gemini client
_google_client = None


def get_google_client():
    """Get or create the Google Gemini client with proper error handling."""
    global _google_client
    if _google_client is None:
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ValueError(
                "google-generativeai package not installed. "
                "Please add 'google-generativeai' to requirements.txt"
            ) from err

        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not google_api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY "
                "as an environment variable."
            )

        try:
            genai.configure(api_key=google_api_key)
            _google_client = genai
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Google Gemini client: {e!s}. "
                "Please verify your GOOGLE_API_KEY is correct."
            ) from e

    return _google_client


# ──────────────────────────────────────────────────────────────
# Load environment variables
load_dotenv()
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_STRING = "gpt-4.1-mini"  # we default on gpt-4.1-mini
openai_api_key = os.getenv("OPENAI_API_KEY")
# print(openai_api_key)  # Do NOT log secrets!
client = openai.OpenAI(api_key=openai_api_key)

# Lazy initialization of bedrock client to avoid errors if credentials are missing
_bedrock_runtime = None


def get_bedrock_client():
    """Get or create the Bedrock runtime client with proper error handling."""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip()

        if not aws_access_key or not aws_secret_key:
            raise ValueError(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "as environment variables. "
                f"Current values: AWS_ACCESS_KEY_ID={'***' if aws_access_key else 'EMPTY'}, "
                f"AWS_SECRET_ACCESS_KEY={'***' if aws_secret_key else 'EMPTY'}"
            )

        try:
            _bedrock_runtime = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize AWS Bedrock client: {e!s}. "
                "Please verify your AWS credentials are valid and have Bedrock access."
            ) from e

    return _bedrock_runtime


# ──────────────────────────────────────────────────────────────
# Model switcher
# ──────────────────────────────────────────────────────────────
def set_model(model_id: str) -> None:
    global MODEL_STRING
    MODEL_STRING = model_id
    print(f"Model changed to: {model_id}")


def set_provider(provider: str) -> None:
    global PROVIDER


# ──────────────────────────────────────────────────────────────
# High-level Chat wrapper
# ──────────────────────────────────────────────────────────────
def chat(messages, persona):
    provider = MODEL_TO_PROVIDER[MODEL_STRING]

    if provider == "openai":
        print("Using openai: ", MODEL_STRING)
        t0 = time.time()

        # Add system prompt for better behavior
        system_prompt = ""

        # Prepare messages with system prompt
        chat_messages = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(
            [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        )

        request_kwargs = {
            "model": MODEL_STRING,
            "messages": chat_messages,
            "max_completion_tokens": 4000,
        }
        # Some newer OpenAI models only support the default temperature.
        if MODEL_STRING not in {"gpt-5", "gpt-5-nano", "gpt-5-mini"}:
            request_kwargs["temperature"] = 0.3

        response = client.chat.completions.create(**request_kwargs)

        dt = time.time() - t0
        text = response.choices[0].message.content.strip()

        # Calculate tokens
        total_tok = response.usage.total_tokens if response.usage else len(text.split())

        return text, dt, total_tok, (total_tok / dt if dt else total_tok)
    if provider == "anthropic":
        print("Using anthropic: ", MODEL_STRING)
        t0 = time.time()

        # Add system prompt for better behavior
        system_prompt = ""

        claude_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        try:
            bedrock_runtime = get_bedrock_client()

            # Use inference profile ARN if available (for provisioned throughput models)
            # Otherwise use modelId (for on-demand models)
            invoke_kwargs = {
                "contentType": "application/json",
                "accept": "application/json",
                "body": json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "system": system_prompt,
                        "messages": claude_messages,
                        "max_tokens": 4000,  # Much higher limit for longer responses
                        "temperature": 0.3,  # Lower temperature for more focused responses
                    }
                ),
            }

            # Check if this model has an inference profile ARN (provisioned throughput)
            # For provisioned throughput, use the ARN as the modelId
            if MODEL_STRING in MODEL_TO_INFERENCE_PROFILE_ARN:
                invoke_kwargs["modelId"] = MODEL_TO_INFERENCE_PROFILE_ARN[MODEL_STRING]
            else:
                invoke_kwargs["modelId"] = MODEL_STRING

            response = bedrock_runtime.invoke_model(**invoke_kwargs)

            dt = time.time() - t0
            body = json.loads(response["body"].read())
        except ValueError:
            # Re-raise ValueError (credential errors) as-is
            raise
        except Exception as e:
            error_msg = str(e)
            if (
                "ValidationException" in error_msg
                and "model identifier is invalid" in error_msg
            ):
                raise ValueError(
                    f"Invalid Bedrock model ID: '{MODEL_STRING}'. "
                    f"Error: {error_msg}. "
                    "Please verify the model ID is correct and the model is available in your AWS region. "
                    "Common Claude model IDs: 'anthropic.claude-3-5-sonnet-20241022-v2' or 'anthropic.claude-3-haiku-20240307-v1'"
                ) from e
            if (
                "UnrecognizedClientException" in error_msg
                or "invalid" in error_msg.lower()
            ):
                raise ValueError(
                    f"AWS Bedrock authentication failed: {error_msg}. "
                    "Please verify your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY secrets "
                    "are correct and have Bedrock access permissions."
                ) from e
            raise

        text = "".join(
            part["text"] for part in body["content"] if part["type"] == "text"
        ).strip()
        total_tok = len(text.split())

        return text, dt, total_tok, (total_tok / dt if dt else total_tok)
    if provider == "google":
        print("Using google (Gemini): ", MODEL_STRING)
        t0 = time.time()

        try:
            genai = get_google_client()

            # Get the model
            model = genai.GenerativeModel(MODEL_STRING)

            # Convert messages to Gemini format
            # Gemini API expects a chat history format with "user" and "model" roles
            chat_history = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Skip system messages (we'll handle them separately)
                if role == "system":
                    continue
                # Gemini uses "model" instead of "assistant"
                if role == "assistant":
                    role = "model"
                chat_history.append({"role": role, "parts": [content]})

            # Separate history from the last user message
            if chat_history and chat_history[-1]["role"] == "user":
                history = chat_history[:-1]
                last_user_message = chat_history[-1]["parts"][0]
            else:
                history = []
                last_user_message = chat_history[-1]["parts"][0] if chat_history else ""

            # Start a chat session with history
            chat = model.start_chat(history=history)

            # Send the last message
            response = chat.send_message(
                last_user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.3,
                ),
            )

            dt = time.time() - t0
            text = response.text.strip()

            # Calculate tokens (approximate)
            total_tok = len(text.split())

            return text, dt, total_tok, (total_tok / dt if dt else total_tok)
        except ValueError:
            # Re-raise ValueError (credential errors) as-is
            raise
        except Exception as e:
            error_msg = str(e)
            if (
                "API key" in error_msg
                or "invalid" in error_msg.lower()
                or "401" in error_msg
                or "403" in error_msg
            ):
                raise ValueError(
                    f"Google API authentication failed: {error_msg}. "
                    "Please verify your GOOGLE_API_KEY secret is correct and has Gemini API access."
                ) from e
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise ValueError(
                    f"Invalid Gemini model ID: '{MODEL_STRING}'. "
                    f"Error: {error_msg}. "
                    "Please verify the model ID is correct. "
                    "Common Gemini model IDs: 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-pro'"
                ) from e
            raise
    elif provider == "deepseek":
        print("Using deepseek: ", MODEL_STRING)
        t0 = time.time()

        system_prompt = ""

        ds_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]
        for msg in messages:
            role = msg.get("role", "user")
            ds_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": msg["content"]}],
                }
            )

        try:
            bedrock_runtime = get_bedrock_client()
            response = bedrock_runtime.invoke_model(
                modelId=MODEL_STRING,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {
                        "messages": ds_messages,
                        "max_completion_tokens": 500,
                        "temperature": 0.5,
                        "top_p": 0.9,
                    }
                ),
            )

            dt = time.time() - t0
            body = json.loads(response["body"].read())
        except ValueError:
            # Re-raise ValueError (credential errors) as-is
            raise
        except Exception as e:
            error_msg = str(e)
            if (
                "ValidationException" in error_msg
                and "model identifier is invalid" in error_msg
            ):
                raise ValueError(
                    f"Invalid Bedrock model ID: '{MODEL_STRING}'. "
                    f"Error: {error_msg}. "
                    "Please verify the model ID is correct and the model is available in your AWS region."
                ) from e
            if (
                "UnrecognizedClientException" in error_msg
                or "invalid" in error_msg.lower()
            ):
                raise ValueError(
                    f"AWS Bedrock authentication failed: {error_msg}. "
                    "Please verify your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY secrets "
                    "are correct and have Bedrock access permissions."
                ) from e
            raise

        outputs = body.get("output", [])
        text_chunks = []
        for item in outputs:
            for content in item.get("content", []):
                chunk_text = content.get("text") or content.get("output_text")
                if chunk_text:
                    text_chunks.append(chunk_text)
        text = "".join(text_chunks).strip()
        if not text and "response" in body:
            text = body["response"].get("output_text", "").strip()
        total_tok = len(text.split())

        return text, dt, total_tok, (total_tok / dt if dt else total_tok)
    return None
    # elif provider == "meta":
    #     print("Using meta (LLaMA): ", MODEL_STRING)
    #     t0 = time.time()

    #     # Add system prompt for better behavior
    #     system_prompt = ""

    #     # Format conversation properly for Llama3
    #     formatted_prompt = "<|begin_of_text|>"

    #     # Add system prompt
    #     formatted_prompt += "<|start_header_id|>system<|end_header_id|>\n" + system_prompt + "<|eot_id|>\n"

    #     # Add conversation history
    #     for msg in messages:
    #         if msg["role"] == "user":
    #             formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n" + msg["content"] + "<|eot_id|>\n"
    #         elif msg["role"] == "assistant":
    #             formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n" + msg["content"] + "<|eot_id|>\n"

    #     # Add final assistant prompt
    #     formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    #     response = bedrock_runtime.invoke_model(
    #         modelId=MODEL_STRING,
    #         contentType="application/json",
    #         accept="application/json",
    #         body=json.dumps(
    #             {
    #                 "prompt": formatted_prompt,
    #                 "max_gen_len": 512,  # Shorter responses
    #                 "temperature": 0.3,  # Lower temperature for more focused responses
    #             }
    #         ),
    #     )

    # dt = time.time() - t0
    # body = json.loads(response["body"].read())
    # text = body.get("generation", "").strip()
    # total_tok = len(text.split())

    #     return text, dt, total_tok, (total_tok / dt if dt else total_tok)
    # elif provider == "mistral":
    #     print("Using mistral: ", MODEL_STRING)
    #     t0 = time.time()

    #     prompt = messages[-1]["content"]
    #     formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    #     response = bedrock_runtime.invoke_model(
    #         modelId=MODEL_STRING,
    #         contentType="application/json",
    #         accept="application/json",
    #         body=json.dumps(
    #             {"prompt": formatted_prompt, "max_tokens": 512, "temperature": 0.5}
    #         ),
    #     )

    #     dt = time.time() - t0
    #     body = json.loads(response["body"].read())

    #     text = body["outputs"][0]["text"].strip()
    #     total_tok = len(text.split())

    #     return text, dt, total_tok, (total_tok / dt if dt else total_tok)
    # elif provider == "ollama":
    #     print("Using ollama: ", MODEL_STRING)
    #     t0 = time.time()

    #     # Format messages for Ollama API with system prompt
    #     ollama_messages = []

    #     # Add system prompt for better behavior
    #     system_prompt = ""
    #     ollama_messages.append({
    #         "role": "system",
    #         "content": system_prompt
    #     })

    #     for msg in messages:
    #         ollama_messages.append({
    #             "role": msg["role"],
    #             "content": msg["content"]
    #         })

    #     # Make request to Ollama API
    #     response = requests.post(
    #         f"{OLLAMA_BASE_URL}/api/chat",
    #         json={
    #             "model": MODEL_STRING,
    #             "messages": ollama_messages,
    #             "stream": False,
    #             "options": {
    #                 "temperature": 0.3,  # Lower temperature for more focused responses
    #                 # "num_predict": 4000,  # Much higher limit for longer responses
    #                 "top_p": 0.9,
    #                 "repeat_penalty": 1.1
    #             }
    #         },
    #         timeout=60
    #     )

    #     dt = time.time() - t0

    #     if response.status_code == 200:
    #         result = response.json()
    #         text = result["message"]["content"].strip()
    #         total_tok = len(text.split())
    #         return text, dt, total_tok, (total_tok / dt if dt else total_tok)
    #     else:
    #         raise Exception(f"Ollama API error: {response.status_code} - {response.text}")


# ──────────────────────────────────────────────────────────────
# Diagnostics / CLI test
# ──────────────────────────────────────────────────────────────
def check_credentials():
    # # Check if using Ollama (no API key required)
    # if MODEL_TO_PROVIDER.get(MODEL_STRING) == "ollama":
    #     # Test Ollama connection
    #     try:
    #         response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    #         if response.status_code == 200:
    #             print("Ollama connection successful")
    #             return True
    #         else:
    #             print(f"Ollama connection failed: {response.status_code}")
    #             return False
    #     except Exception as e:
    #         print(f"Ollama connection failed: {e}")
    #         return False

    # Check if using Bedrock providers (anthropic, meta, mistral, deepseek)
    bedrock_providers = ["anthropic"]
    if MODEL_TO_PROVIDER.get(MODEL_STRING) in bedrock_providers:
        # Test AWS Bedrock connection by trying to invoke a simple model
        try:
            bedrock_runtime = get_bedrock_client()
            # Try a simple test invocation to verify credentials
            test_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
            test_kwargs = {
                "contentType": "application/json",
                "accept": "application/json",
                "body": json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 10,
                        "temperature": 0.1,
                    }
                ),
            }

            # Use inference profile ARN if available (use ARN as modelId for provisioned throughput)
            if test_model in MODEL_TO_INFERENCE_PROFILE_ARN:
                test_kwargs["modelId"] = MODEL_TO_INFERENCE_PROFILE_ARN[test_model]
            else:
                test_kwargs["modelId"] = test_model

            bedrock_runtime.invoke_model(**test_kwargs)
            print("Bedrock connection successful")
            return True
        except Exception as e:
            print(f"Bedrock connection failed: {e}")
            print(
                "Make sure AWS credentials are configured and you have access to Bedrock"
            )
            return False

    # For OpenAI, check API key
    if MODEL_TO_PROVIDER.get(MODEL_STRING) == "openai":
        required = ["MODEL_API_KEY"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            print(f"Missing environment variables: {missing}")
            return False
        return True

    # For Google Gemini, check API key
    if MODEL_TO_PROVIDER.get(MODEL_STRING) == "google":
        required = ["GOOGLE_API_KEY"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            print(f"Missing environment variables: {missing}")
            return False
        # Try to initialize the client to verify the key works
        try:
            get_google_client()
            return True
        except Exception as e:
            print(f"Google API client initialization failed: {e}")
            return False

    return True


def test_chat():
    print("Testing chat...")
    try:
        test_messages = [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'Test successful'.",
            }
        ]
        text, latency, tokens, tps = chat(test_messages)
        print(f"Test passed!  {text}  {latency:.2f}s  {tokens} ⚡ {tps:.1f} tps")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    print("running diagnostics")
    if check_credentials():
        test_chat()
    print("\nDone.")

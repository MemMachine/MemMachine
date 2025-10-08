import os
import re

import requests


class OpenAISummary:
    def __init__(self, api_key, model=None):
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        if not model:
            self.model = "gpt-4.1-mini"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.openai_session = requests.Session()
        self.openai_session.headers.update(self.headers)
        self.temperature = 0.7
        self.max_tokens = 150
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop = None

    def list_models(self):
        response = requests.get(
            "https://api.openai.com/v1/models", headers=self.headers
        )
        return response.json()

    def get_memory_summary_prompt(self, text):
        return f"Please summarize the following conversation messages:\n\n{text}"

    def summarize(self, text):
        content = self.get_memory_summary_prompt(text)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }
        response = self.openai_session.post(self.openai_url, json=payload, timeout=300)
        self.openai_session.close()
        return response.json()


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        print(
            "Error: API key not found, please set environment variable OPENAI_API_KEY"
        )
        exit(1)

    openai_summary = OpenAISummary(api_key=api_key)
    # models = openai_summary.list_models()
    # print(models)
    # exit(0)

    # Read messages from extracted file
    messages = []
    with open("extracted/locomo10_extracted_20250930140245_conv_1.txt", "r") as f:
        for line in f:
            line = line.strip()
            line = re.sub(r"\\n", " ", line)
            line = re.sub(r"\n", " ", line)
            messages.append(line)

    print(f"Total messages loaded: {len(messages)}")

    # Process messages in batches of 20
    batch_size = 20
    batch_num = 1

    for i in range(0, len(messages), batch_size):
        batch = messages[i : i + batch_size]
        print(
            f"\n--- Processing Batch {batch_num} (messages {i + 1}-{min(i + batch_size, len(messages))}) ---"
        )

        # Join messages for summarization
        batch_text = "\n".join(batch)

        # Create summary prompt
        summary_prompt = (
            f"Please summarize the following conversation messages:\n\n{batch_text}"
        )

        try:
            # Get summary from OpenAI
            response = openai_summary.summarize(summary_prompt)

            if "choices" in response and len(response["choices"]) > 0:
                summary = response["choices"][0]["message"]["content"]
                print(f"Batch {batch_num} Summary:")
                print(summary)
            else:
                print(f"Error: No summary generated for batch {batch_num}")
                print(f"Response: {response}")

        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")

        batch_num += 1

    print(f"\nCompleted processing {batch_num - 1} batches")

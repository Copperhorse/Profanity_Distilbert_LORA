import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import time

from transformers import AutoModelForCausalLM, AutoTokenizer

max_retries = 10
retry_count = 0

print("Starting download with automatic retries...")
print("Model size: 2.34GB")
print("This will retry automatically if your connection drops.\n")

while retry_count < max_retries:
    try:
        print(f"Attempt {retry_count + 1}/{max_retries}")

        model = AutoModelForCausalLM.from_pretrained(
            "LiquidAI/LFM2.5-1.2B-Instruct",
            trust_remote_code=True,
            resume_download=True,  # Important for resuming
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "LiquidAI/LFM2.5-1.2B-Instruct",
            resume_download=True,
        )

        print("\n✓ Download complete!")
        break

    except Exception as e:
        retry_count += 1
        if retry_count < max_retries:
            wait_time = 10
            print(f"\n✗ Download failed: {str(e)[:100]}")
            print(f"Waiting {wait_time}s before retry {retry_count + 1}...\n")
            time.sleep(wait_time)
        else:
            print(
                "\n✗ All retries exhausted. Try again later or from a better network."
            )
            raise

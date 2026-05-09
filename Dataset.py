import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_path = "/home/copper/Work/Fine_tuningDistilBert/lfm-profanity-lora"
base_model_name = "LiquidAI/LFM2.5-1.2B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)

print("Downloading/loading base model (this may take a few minutes on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,  # Changed from torch_dtype (deprecated)
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)

print("Model loaded successfully!")


# this is the code for prompting:
def hybrid_analyze_review(review_text):
    # ====================================================
    # STEP 1: CLASSIFICATION (Adapter ON)
    # ====================================================
    detect_prompt = f"""<|im_start|>system
You are a helpful AI assistant specialized in content moderation.<|im_end|>
<|im_start|>user
Analyze the following customer review and provide:
1. Whether it contains profanity (yes/no)
2. The sentiment (positive/negative/neutral)

Review: {review_text}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer([detect_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
    analysis = (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .split("assistant")[-1]
        .strip()
    )

    # Parse logic
    has_profanity = "Profanity: yes" in analysis
    is_positive = "Sentiment: positive" in analysis.lower()

    # ====================================================
    # STEP 2: REWRITE (Base Model - Adapter OFF)
    # ====================================================
    if has_profanity:
        if is_positive:
            # POSITIVE CASE
            system_task = (
                "You are a text rewriting engine. "
                "Your task is to remove profanity from enthusiastic praise while preserving excitement."
            )

            constraints = (
                "Transformation rules:\n"
                "- Preserve the original enthusiasm and positive emphasis.\n"
                "- Replace profanity with strong but professional praise.\n"
                "- Do NOT weaken the sentiment.\n"
                "- Do NOT summarize or neutralize excitement.\n"
                "- Output only the rewritten review text."
            )

            examples = """
          Original: Holy shit, this espresso machine is amazing!
          Rewritten: This espresso machine is absolutely amazing!

          Original: This game is fucking incredible. I can't put it down.
          Rewritten: This game is incredibly engaging. I cannot put it down.

          Original: Hell yeah! Arrived in one day. You guys are kickass.
          Rewritten: Excellent! It arrived in one day and the service was outstanding.
          """
            temperature = 0.6

        else:
            # NEGATIVE CASE
            system_task = (
                "You are a text rewriting engine. "
                "Your task is to remove profanity and insults while preserving the original complaint exactly."
            )

            constraints = (
                "Transformation rules:\n"
                "- Perform minimal rewriting. Replace only the profane or insulting words.\n"
                "- Preserve all concrete details such as actions, timelines, failures, and outcomes.\n"
                "- Do NOT summarize, generalize, or abstract the complaint.\n"
                "- Do NOT remove mentions of money, delays, damage, or lack of response.\n"
                "- Maintain a professional but firm tone.\n"
                "- Output only the rewritten review text."
            )

            examples = """
          Original: This laptop is a piece of shit. It broke after two days. What the hell?
          Rewritten: This laptop is unacceptable. It broke after two days.

          Original: Don't buy from this bastard seller. They took my money and ghosted me.
          Rewritten: Do not buy from this seller. They took my money and stopped responding.

          Original: The food tasted like crap and the waiter was a total ass.
          Rewritten: The food tasted poor and the waiter was unprofessional.
          """
            temperature = 0.3

        # -------- PROMPT ASSEMBLY --------
        rewrite_prompt = f"""<|im_start|>system
    {system_task}

    {constraints}
    <|im_end|>
    <|im_start|>user
    Below are examples of correct rewrites:

    {examples}

    Rewrite the following review.

    Original:
    {review_text}

    Rewritten:
    <|im_end|>
    <|im_start|>assistant
    """

        inputs = tokenizer([rewrite_prompt], return_tensors="pt").to("cuda")

        with model.disable_adapter():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=temperature,
                do_sample=True,
            )

        rewrite = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -------- SAFE CLEANUP --------
        rewrite = rewrite.split("assistant")[-1].strip()
        rewrite = rewrite.replace("Rewritten:", "").strip()
        rewrite = rewrite.splitlines()[0]

        return f"{analysis}\nRewritten: {rewrite}"

    else:
        return f"{analysis}\nRewritten: Not needed"

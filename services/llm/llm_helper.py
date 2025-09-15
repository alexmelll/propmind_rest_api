import os
import json
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_llm_config(config_filename="llm_config.json"):
    full_path = os.path.join(BASE_DIR, config_filename)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompts(prompts_filename="llm_prompt.yaml"):
    full_path = os.path.join(BASE_DIR, prompts_filename)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def explain_price_factors(price_explain: dict,
                          prompt_name: str = "price_explain",
                          config_path="llm_config.json",
                          prompts_path="llm_prompt.yaml") -> str:
    """
    Given a dict of model explanation results, returns a natural-language explanation.
    """

    # Load config + prompts
    config = load_llm_config(config_path)
    prompts = load_prompts(prompts_path)

    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in {prompts_path}")

    # Fill in structured_info
    structured_info = json.dumps(price_explain, indent=2, default=str)
    system_prompt = prompts[prompt_name]["system"]
    system_prompt = system_prompt.replace("{{structured_info}}", structured_info)

    # Init client
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_API_ENDPOINT")
    if not api_key or not base_url:
        raise ValueError("Set QWEN_API_KEY and QWEN_API_ENDPOINT environment variables.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Call LLM
    completion = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        stream=config.get("stream"),
        extra_body={
            "enable_thoughts": config.get("enable_thoughts", False),
            "thought_budget": config.get("thought_budget", 0)
        }
    )

    if config.get("stream", False):
        answer = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                # print(delta.content, end="", flush=True)
                answer += delta.content

        return answer.strip()
    else:
        return completion.choices[0].message.content.strip()
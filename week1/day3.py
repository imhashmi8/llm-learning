"""
Technical question explainer using GPT-4o-mini (streaming) and Llama 3.2 via Ollama.

Week 1 Exercise: Build a tool that takes a technical question and explains it.

Usage:
    uv run tech-explainer "your question here"
    uv run tech-explainer  # prompts interactively
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL_GPT = "gpt-4o-mini"
MODEL_LLAMA = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

SYSTEM_PROMPT = (
    "You are a helpful assistant that explains technical programming and computer science concepts clearly. "
    "Provide step-by-step explanations with examples where helpful. Respond in markdown."
)


def explain_with_gpt(question: str) -> None:
    """Stream an explanation from GPT-4o-mini."""
    client = OpenAI()
    print("\n--- GPT-4o-mini ---\n")
    stream = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def explain_with_llama(question: str) -> None:
    """Get an explanation from Llama 3.2 via Ollama."""
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    print("\n--- Llama 3.2 (Ollama) ---\n")
    response = ollama.chat.completions.create(
        model=MODEL_LLAMA,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    print(response.choices[0].message.content)


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your technical question: ").strip()
        if not question:
            print("No question provided.")
            sys.exit(1)

    print(f"\nQuestion: {question}")

    explain_with_gpt(question)
    explain_with_llama(question)


if __name__ == "__main__":
    main()

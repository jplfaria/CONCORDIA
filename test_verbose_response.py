#!/usr/bin/env python3
import os
import logging
from concord.llm.argo_gateway import ArgoGatewayClient

# Set up verbose logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("verbose_test")

def main():
    # Check environment setup
    if "ARGO_USER" not in os.environ:
        print("ERROR: ARGO_USER environment variable not set")
        print("Set it with: export ARGO_USER=<your-username>")
        return

    # Create client with GPT-4o model (correct format)
    client = ArgoGatewayClient(model="gpt4o")
    
    # Complex biomedical terms to test
    term_a = "Parkinson's disease"
    term_b = "Lewy body dementia"
    
    # Detailed prompt that should produce a more verbose response
    prompt = f"""You are a biomedical entity relationship expert.
    
Your task is to classify the relationship between these two biomedical entities:

A: {term_a}
B: {term_b}

Classify their relationship using one of the following labels:
- Exact: Entities are identical or functionally equivalent
- Synonym: Different terms for the same concept
- Broader: A is a broader concept than B
- Narrower: A is a narrower concept than B
- Related: Entities are related but don't fit the above categories
- Uninformative: Not enough information to determine relationship
- Different: Entities are completely different concepts

Analyze carefully. Return your answer as: **<Label> — <detailed explanation>**
"""
    
    print(f"\nSending request to classify: {term_a} vs {term_b}")
    print(f"Using model: {client.model}")
    
    # Get raw response
    try:
        raw_response = client.chat(prompt)
        print("\n=== Raw LLM Response ===")
        print(raw_response)
        print("=========================\n")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Parse the response manually using similar logic to _parse function
    if not raw_response:
        print("No response received")
        return
        
    raw = raw_response.strip().rstrip(".")
    import re
    dash_pattern = re.compile(r"\s*[—–-]\s*")
    parts = dash_pattern.split(raw, maxsplit=1)
    
    label_part = parts[0].strip().replace('*', '').strip()
    label = label_part.split()[0].capitalize()
    note = parts[1].strip() if len(parts) > 1 else ""
    
    print("=== Parsed Result ===")
    print(f"Label: {label}")
    print(f"Evidence: {note}")

if __name__ == "__main__":
    main()
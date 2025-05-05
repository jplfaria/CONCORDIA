#!/usr/bin/env python3
# Save as debug_llm_call.py

import os
import logging
from concord.llm.argo_gateway import ArgoGatewayClient, llm_label
from concord.llm.prompts import build_annotation_prompt, get_prompt_template

# Set detailed logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')

def main():
    # Check for ARGO_USER
    if "ARGO_USER" not in os.environ:
        print("ERROR: ARGO_USER environment variable not set")
        return

    # Create client with the correct model name format
    client = ArgoGatewayClient(model="gpt4o")
    
    # Terms to compare
    text_a = "Glucose phosphate isomerase"
    text_b = "Phosphoglucose isomerase"
    
    print(f"\nTesting entity relationship classification:")
    print(f"A: {text_a}")
    print(f"B: {text_b}")
    print(f"Model: {client.model}")
    
    # Get the template directly
    template = get_prompt_template({"prompt_ver": "v1.3-test"})
    print(f"\nUsing template version: v1.3-test")
    
    # Build the prompt manually to see what's being sent
    prompt = build_annotation_prompt(text_a, text_b, template)
    print("\nFull prompt being sent to LLM:")
    print("------------------------")
    print(prompt)
    print("------------------------")
    
    # Make the direct call to the LLM
    print("\nMaking direct call to Argo Gateway...")
    try:
        # Send with system message explicitly
        system_msg = "You are a bioinformatics assistant specializing in entity relationships. Respond with '<Label> — <detailed explanation of the relationship with specific evidence>'."
        raw_response = client.chat(prompt, system=system_msg)
        
        print("\nRaw LLM Response:")
        print("------------------------")
        print(raw_response)
        print("------------------------")
        
        # Let's also use the llm_label function to see what it returns
        label, note = llm_label(text_a, text_b, client, template=template, with_note=True)
        
        print("\nProcessed result from llm_label function:")
        print(f"Label: {label}")
        print(f"Evidence: {note}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
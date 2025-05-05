#!/usr/bin/env python3
# Test script for Argo Gateway API

import os
import sys
import logging
import requests
import json
from concord.llm.argo_gateway import ArgoGatewayClient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("test_argo")

def test_simple_request():
    """Test a very simple request to the Argo Gateway"""
    print("Testing Argo Gateway with minimal prompt")
    
    # Create client with default settings
    try:
        # Note: model should be gpto3mini without hyphen according to docs
        client = ArgoGatewayClient(model="gpto3mini")
        
        # Simple prompt with no formatting requirements
        simple_prompt = "Please respond with 'Hello World' to confirm you're working."
        
        print(f"Connecting to Argo Gateway with model: {client.model}")
        if hasattr(client, 'base_url'):
            print(f"API URL: {client.base_url}")
        if hasattr(client, 'user'):
            print(f"User: {client.user}")
        
        # Try sending the request
        print("\nSending request...")
        response = client.chat(simple_prompt)
        
        # Print response
        print("\nResponse received:")
        print(response)
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError testing Argo Gateway: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response headers: {e.response.headers}")
            print(f"Response content: {e.response.content}")
        return False

def test_valid_models():
    """Test with the correct model names from the documentation"""
    print("\n\nTesting Argo Gateway with valid model names")
    
    # List of valid models from the documentation
    valid_models = ["gpt35", "gpt4", "gpt4o", "gpt4turbo"]
    
    for model in valid_models:
        try:
            print(f"\nTesting model: {model}")
            client = ArgoGatewayClient(model=model)
            
            # Simple prompt with no formatting requirements
            simple_prompt = "Please respond with 'Hello World' to confirm you're working."
            
            # Try sending the request
            print(f"Sending request to {model}...")
            response = client.chat(simple_prompt)
            
            # Print response
            print(f"Response from {model}:")
            print(response)
            print(f"{model} test successful!")
            
        except Exception as e:
            print(f"Error testing {model}: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response headers: {e.response.headers}")
                print(f"Response content: {e.response.content}")

def test_direct_api_call():
    """Test by directly calling the API without using the client class"""
    print("\n\nTesting direct API call to Argo Gateway")
    
    try:
        # API endpoint to POST
        url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
        
        # Check for ARGO_USER environment variable
        user = os.environ.get("ARGO_USER", "user_not_set")
        
        # Data according to documentation
        data = {
            "user": user,
            "model": "gpt4o",  # Correct format without hyphen
            "system": "You are a helpful assistant.",
            "prompt": ["Please respond with 'Hello World' to confirm you're working."],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 100
        }
        
        # Convert to JSON
        payload = json.dumps(data)
        
        # Set content type
        headers = {"Content-Type": "application/json"}
        
        print(f"Sending direct API request to {url}")
        print(f"Request payload: {payload}")
        
        # Send POST request
        response = requests.post(url, data=payload, headers=headers)
        
        # Print response
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Direct API call successful!")
        else:
            print("Direct API call failed.")
            
    except Exception as e:
        print(f"Error with direct API call: {e}")
        print(f"Error type: {type(e).__name__}")

def test_gpt4o_request():
    """Test the GPT-4o model specifically"""
    print("\n\nTesting Argo Gateway with GPT-4o model")
    
    try:
        client = ArgoGatewayClient(model="gpt4o")
        
        # Simple prompt with no formatting requirements
        simple_prompt = "Please respond with 'Hello World' to confirm you're working."
        
        print(f"Connecting to Argo Gateway with model: {client.model}")
        
        # Try sending the request
        print("\nSending request...")
        response = client.chat(simple_prompt)
        
        # Print response
        print("\nResponse received:")
        print(response)
        
        print("\nGPT-4o test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError testing Argo Gateway with GPT-4o: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response headers: {e.response.headers}")
            print(f"Response content: {e.response.content}")
        return False

if __name__ == "__main__":
    print("=== Argo Gateway Connection Test ===\n")
    
    # Check environment variables
    if "ARGO_USER" not in os.environ:
        print("Warning: ARGO_USER environment variable not set")
        print("Try setting it with: export ARGO_USER=<your-username>")
        print("This is required for the Argo Gateway API according to the documentation\n")
    
    # Run tests
    test_simple_request()
    test_valid_models()
    test_direct_api_call()
    test_gpt4o_request()
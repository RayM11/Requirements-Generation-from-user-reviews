import os
import time
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


class LLMManager:
    """
    A manager class for interacting with Language Learning Models through
    Hugging Face Inference Providers API.

    This class provides a clean interface for initializing connections to various
    LLM providers and making inference requests with retry logic and error handling.
    """

    # Default configuration values
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324"
    DEFAULT_PROVIDER = "fireworks-ai"
    DEFAULT_GENERATION_PARAMS = {
        "max_tokens": 3000,
        "temperature": 0.5,
        "top_p": 0.90,
        "stream": False
    }

    def __init__(
            self,
            model_name: str = None,
            provider: str = None,
            generation_params: Dict[str, Any] = None,
            hf_token: str = None,
            test_connection: bool = True
    ):
        """
        Initialize the LLM Manager with specified configuration.

        Args:
            model_name (str, optional): Name of the model to use.
                                      Defaults to DEFAULT_MODEL.
            provider (str, optional): Inference provider name.
                                    Defaults to DEFAULT_PROVIDER.
            generation_params (dict, optional): Parameters for text generation.
                                              Defaults to DEFAULT_GENERATION_PARAMS.
            hf_token (str, optional): Hugging Face API token. If None, will try to
                                    get from HF_TOKEN environment variable.
            test_connection (bool): Whether to test the API connection on initialization.
                                  Defaults to True.

        Raises:
            ValueError: If HF_TOKEN is not provided and not found in environment variables.
            Exception: If API connection test fails (when test_connection=True).
        """
        # Set model and provider with defaults
        self.model_name = model_name or self.DEFAULT_MODEL
        self.provider = provider or self.DEFAULT_PROVIDER

        # Set generation parameters with defaults
        self.generation_params = self.DEFAULT_GENERATION_PARAMS.copy()
        if generation_params:
            self.generation_params.update(generation_params)

        # Get HF token from parameter or environment variable
        load_dotenv()
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN not provided. Please provide it as parameter or set the "
                "HF_TOKEN environment variable."
            )

        # Initialize the InferenceClient
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.hf_token
        )

        # Test connection if requested
        if test_connection:
            self._test_api_connection()

    def _test_api_connection(self) -> bool:
        """
        Test if the model and provider are accessible via the API.

        Returns:
            bool: True if connection is successful.

        Raises:
            Exception: If the API connection fails.
        """
        print(f"Testing connection to model: {self.model_name}")
        print(f"Using provider: {self.provider}")

        try:
            # Simple test request
            test_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": "Hello, can you respond with 'OK'?"}
                ],
                max_tokens=10,
                temperature=0.1
            )

            response = test_completion.choices[0].message.content
            print("✅ API connection successful!")
            print(f"Test response: {response[:50]}...")
            return True

        except Exception as e:
            print(f"❌ API connection failed: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Check if your HF token has 'inference.serverless.write' scope")
            print("2. Verify the model is available on the provider")
            print("3. Some models may require approval or PRO subscription")
            print(f"4. Check model availability at: https://huggingface.co/{self.model_name}")
            raise

    def update_generation_params(self, **kwargs) -> None:
        """
        Update the generation parameters for future requests.

        Args:
            **kwargs: Key-value pairs of generation parameters to update.
                     Common parameters: max_tokens, temperature, top_p, stream
        """
        self.generation_params.update(kwargs)
        print(f"Updated generation parameters: {self.generation_params}")

    def _generate_with_retry(
            self,
            messages: List[Dict[str, str]],
            max_retries: int = 3
    ) -> str:
        """
        Generate text with retry logic for rate limits or temporary failures.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries for chat completion.
                                           Each dict should have 'role' and 'content' keys.
            max_retries (int): Maximum number of retries. Defaults to 3.

        Returns:
            str: Generated text response from the model.
        """
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.generation_params
                )

                return completion.choices[0].message.content

            except Exception as e:
                error_msg = str(e).lower()

                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = (attempt + 1) * 10
                    print(f"Rate limit hit. Waiting {wait_time} seconds... "
                          f"(Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

                elif "503" in error_msg or "service unavailable" in error_msg:
                    wait_time = 20
                    print(f"Service temporarily unavailable. Waiting {wait_time} seconds... "
                          f"(Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

                else:
                    print(f"Generation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return f"Error after {max_retries} attempts: {str(e)}"

        return "Error: Maximum retries exceeded"

    def generate_response(
            self,
            prompt: str,
            system_message: Optional[str] = None,
            max_retries: int = 3
    ) -> str:
        """
        Generate a response from the LLM given a prompt.

        This is the main function for making requests to the LLM. It handles
        message formatting, retry logic, and error handling.

        Args:
            prompt (str): The main prompt/question to send to the model.
            system_message (str, optional): System message to set context/behavior.
                                          If None, only user message will be sent.
            max_retries (int): Maximum number of retries for failed requests. Defaults to 3.

        Returns:
            str: The model's response to the prompt.

        Example:
            >> llm = LLMManager()
            >> response = llm.generate_response("What is machine learning?")
            >> print(response)
        """
        # Prepare messages for chat completion
        messages = []

        # Add system message if provided
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })

        print(f"Generating response using:")
        print(f"  Model: {self.model_name}")
        print(f"  Provider: {self.provider}")
        print(f"  Prompt length: {len(prompt)} characters")

        # Generate response with retry logic
        response = self._generate_with_retry(messages, max_retries)

        return response

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model configuration.

        Returns:
            dict: Dictionary containing model name, provider, and generation parameters.
        """
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "generation_params": self.generation_params.copy()
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the LLMManager class.
    """

    try:
        # Initialize with default settings
        print("Initializing LLM Manager with default settings...")
        llm = LLMManager()

        # Example 1: Simple prompt
        print("\n" + "=" * 60)
        print("EXAMPLE 1: Simple prompt")
        print("=" * 60)

        simple_prompt = "Explain what artificial intelligence is in one sentence."
        response1 = llm.generate_response(simple_prompt)
        print(f"Response: {response1[:200]}...")

        # Show model info
        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        info = llm.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error in example usage: {str(e)}")
        print("\nMake sure to:")
        print("1. Set your HF_TOKEN environment variable")
        print("2. Install required dependencies: pip install huggingface_hub")
        print("3. Check your internet connection")

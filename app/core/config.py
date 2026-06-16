import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Central configuration hub for Sentinel.
    Fails fast if an API key is missing to prevent silent errors in production.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

    @classmethod
    def validate_keys(cls):
        
        if not cls.GOOGLE_API_KEY:
            raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY is missing.")

        if not cls.PINECONE_API_KEY or "pinecone.io" in cls.PINECONE_API_KEY:
            raise ValueError("CRITICAL ERROR: PINECONE_API_KEY is missing or looks like a URL.")
       
        if not cls.NVIDIA_API_KEY:
            raise ValueError("CRITICAL ERROR: NVIDIA_API_KEY is missing.")

        print("✅ Security Check: All API keys loaded successfully.")

settings = Settings()
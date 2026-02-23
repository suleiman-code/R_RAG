import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Listing all models and their supported methods...")
with open("available_models.txt", "w") as f:
    try:
        models = genai.list_models()
        for m in models:
            f.write(f"Model: {m.name}\n")
            f.write(f"  Display Name: {m.display_name}\n")
            f.write(f"  Description: {m.description}\n")
            f.write(f"  Methods: {m.supported_generation_methods}\n")
            f.write("-" * 20 + "\n")
        print("Model list saved to available_models.txt")
    except Exception as e:
        print(f"Error: {e}")
        f.write(f"Error: {e}\n")

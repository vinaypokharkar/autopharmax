import sys
import os

# Add project root to python path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Project Root Main Entry Point")
    # You can add CLI commands here to run training or backend

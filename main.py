"""Main entry point for Mini AI Knowledge Assistant.

Used in Phase 1 to verify that the environment, directory structure,
and configuration loading are functioning properly.
"""

import sys
import platform
from src.config import get_config_summary


def main():
    print("=" * 60)
    print("  Mini AI Knowledge Assistant - Environment Verification")
    print("=" * 60)
    print(f"Python Version : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Platform       : {platform.system()} {platform.release()}")
    print("-" * 60)

    config = get_config_summary()
    print("Configuration Status:")
    print(f"  Project Root        : {config['base_directory']}")
    print(f"  Data Directory      : {config['data_directory']}")
    print(f"  Gemini API Key Set  : {'Yes' if config['gemini_api_key_set'] else 'No (Placeholder or missing)'}")
    print(f"  Gemini Model        : {config['gemini_model']}")
    print(f"  Embedding Model     : {config['gemini_embedding_model']}")
    print(f"  Default Chunk Size  : {config['default_chunk_size']} characters")
    print(f"  Default Overlap     : {config['default_chunk_overlap']} characters")
    print(f"  Default Top-K       : {config['default_top_k']}")
    print("-" * 60)

    if not config["gemini_api_key_set"]:
        print("[INFO] GEMINI_API_KEY is not configured with an active key.")
        print("       Set your key in .env before running Phase 3+ (Embeddings & Generation).")
    else:
        print("[SUCCESS] Active GEMINI_API_KEY detected.")

    print("\nPhase 1 (Project Setup) verification completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

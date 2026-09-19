"""Basic tests to verify Phase 1 setup and configuration loading."""

import unittest
from pathlib import Path
from src.config import BASE_DIR, DATA_DIR, get_config_summary


class TestPhase1Setup(unittest.TestCase):
    def test_directories_exist(self):
        """Verify that the project directories exist."""
        self.assertTrue(BASE_DIR.exists(), "Base directory does not exist.")
        self.assertTrue(DATA_DIR.exists(), "Data directory does not exist.")

    def test_config_summary_types(self):
        """Verify that configuration settings are parsed into proper types."""
        config = get_config_summary()
        self.assertIsInstance(config["default_chunk_size"], int)
        self.assertIsInstance(config["default_chunk_overlap"], int)
        self.assertIsInstance(config["default_top_k"], int)
        self.assertIsInstance(config["gemini_model"], str)
        self.assertIsInstance(config["gemini_embedding_model"], str)


if __name__ == "__main__":
    unittest.main()

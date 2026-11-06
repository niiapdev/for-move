import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

WORK_DIR = BASE_DIR / 'video_for_whisper'
OUTPUT_DIR = BASE_DIR / 'video_subtitled'
DB_PATH = BASE_DIR / 'jobs.sqlite3'

VK_TOKEN = 'vk1.a.6zeqHlvXb6PXy9kCeXB5pxFoOBfc7d0dZtqKXVZrXYkHyUSe1P_JcY9Kfin5W5EM7WPG6mf0M9OeV8FkO_dG0pgglU4zKyKw3qCzBqPf6OrexKY67FBW3RMMvHM3jw7K5SuCBgwrS0-v6D_gR5v9YEMRMZjE6jtCx2WKOQgqrKEVAMx1bI-SyraMOjcEl2hFd2iM8BtR2ursUyoZVqctoA'

WHISPER_MODEL = 'medium'
WHISPER_DEVICE = 'cpu'
WHISPER_COMPUTE_TYPE = 'int8'
WHISPER_WORD_TIMESTAMPS = True
WHISPER_VAD = True
SUBTITLE_SHIFT_SECONDS = 0.0

BURN_SUBTITLES = True

WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
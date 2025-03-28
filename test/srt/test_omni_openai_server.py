"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import base64
import io
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
import requests
from PIL import Image
from sglatest.srt.test_vision_openai_server import TestOpenAIVisionServer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


## Omni Models
class TestOpenAIOmniServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "minicpmo",
                "--mem-fraction-static",
                "0.7",
                "--tp=2",
            ],
        )
        cls.base_url += "/v1"

    def prepare_audio_messages(self, prompt, audio_file_name):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"{audio_file_name}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        return messages

    def get_audio_response(self, url: str, prompt, category):
        audio_file_path = self.get_or_download_file(url)
        client = openai.Client(api_key="sk-123456", base_url=self.base_url)

        messages = self.prepare_audio_messages(prompt, audio_file_path)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
        )

        audio_response = response.choices[0].message.content

        print("-" * 30)
        print(f"audio {category} response:\n{audio_response}")
        print("-" * 30)

        audio_response = audio_response.lower()

        self.assertIsNotNone(audio_response)
        self.assertGreater(len(audio_response), 0)

        return audio_response

    def _test_audio_speech_completion(self):
        # a fragment of Trump's speech
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            "I have an audio sample. Please repeat the person's words",
            category="speech",
        )
        assert "thank you" in audio_response
        assert "it's a privilege to be here" in audio_response
        assert "leader" in audio_response
        assert "science" in audio_response
        assert "art" in audio_response

    def _test_audio_ambient_completion(self):
        # bird song
        audio_response = self.get_audio_response(
            AUDIO_BIRD_SONG_URL,
            "Please listen to the audio snippet carefully and transcribe the content.",
            "ambient",
        )
        assert "bird" in audio_response

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()


class TestQwen25_oServer(TestOpenAIOmniServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-Omni-7B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "qwen2-5-o",
            ],
        )
        cls.base_url += "/v1"


if __name__ == "__main__":
    unittest.main()

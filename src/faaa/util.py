# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import base64
import hashlib


def generate_id(input_string: str) -> str:
    """
    Generates a unique ID based on the SHA-256 hash of the input string, encoded in a URL-safe base64 format.

    Args:
        input_string (str): The input string to be hashed and encoded.

    Returns:
        str: A URL-safe base64 encoded string representing the SHA-256 hash of the input string.
    """
    hash_bytes = hashlib.sha256(input_string.encode()).digest()
    return base64.urlsafe_b64encode(hash_bytes).decode()

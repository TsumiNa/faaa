# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import base64
import hashlib

import yaml
from pydantic import BaseModel


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


def pydantic_to_yaml(pydantic_obj: BaseModel) -> str:
    """
    Converts a Pydantic object to a YAML-formatted string without brackets or quotes.

    Args:
        pydantic_obj (BaseModel): The Pydantic object to convert.

    Returns:
        str: A YAML-formatted string representing the Pydantic object.
    """
    if not isinstance(pydantic_obj, BaseModel):
        raise ValueError("Input must be a Pydantic BaseModel object.")

    # Convert the Pydantic object to a dictionary
    data = pydantic_obj.model_dump()

    # Convert the dictionary to a YAML-formatted string
    yaml_output = yaml.dump(data, sort_keys=False, default_flow_style=False)

    return yaml_output

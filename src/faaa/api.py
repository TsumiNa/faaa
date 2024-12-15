# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import os

import requests


def make_openrouter_request(messages, tools=None, tool_choice=None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENROUTER_API_KEY environment variable.")

    url = "https://api.openrouter.com/v1/request"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"messages": messages, "tools": tools, "tool_choice": tool_choice}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()

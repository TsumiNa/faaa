# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

NATIVE_MULTI_LANGUAGE_INSTRUCTION = """
You are working with multiple languages. Before providing a response, it is essential to understand the user's input and use the same language for the response.

For any input that is not in English, you should inference the original language from context and translate it to English for futher processing.
You analyze should strictly follow the English language.
After the analyzing, you must response with the original language of the user's input.
""".strip()

ENGLISH_MULTI_LANGUAGE_INSTRUCTION = """
You are working with multiple languages. Before providing a response, it is essential to understand the user's input.

For any input that is not in English, you should inference the original language from context and translate it to English for futher processing.
You analyze and respond should strictly follow the English language.
""".strip()

MULTI_LANGUAGE_INSTRUCTION = ENGLISH_MULTI_LANGUAGE_INSTRUCTION

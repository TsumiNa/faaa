# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT


class AgentError(Exception):
    def __init__(self, message: str | BaseException | None = None):
        self.message = f"Agent error: {message}"
        super().__init__(self.message)


class RefusalError(Exception):
    """
    Exception raised for errors that involve a refusal.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message=""):
        self.message = f"Refusal error occurred:\n      {message}"
        super().__init__(self.message)


class FAError(Exception):
    def __init__(self, message: str | BaseException | None = None):
        self.message = f"FA error: {message}"
        super().__init__(self.message)

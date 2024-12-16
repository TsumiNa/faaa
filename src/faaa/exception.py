# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT


class RefusalError(Exception):
    """
    Exception raised for errors that involve a refusal.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message=""):
        self.message = f"Refusal error occurred:\n      {message}"
        super().__init__(self.message)

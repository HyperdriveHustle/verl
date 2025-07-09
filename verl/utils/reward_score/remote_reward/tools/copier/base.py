from typing import List, Dict

from harpy.tools.base import Tool, GenerateTool


class CopyTool(GenerateTool):
    def __init__(self, name: str, **kwargs):
        self.name = name

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        return [messages[-1]["content"]]

    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs) -> List[List[str]]:
        return [[messages[-1]["content"]] for messages in batch_messages]
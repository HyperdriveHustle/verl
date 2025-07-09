from typing import Any, List, Dict


class Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class GenerateTool(Tool):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(name)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        raise NotImplementedError

    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs) -> List[List[str]]:
        raise NotImplementedError

class SortTool(Tool):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(name)
    
    def sort(self, responses: List[str], messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError
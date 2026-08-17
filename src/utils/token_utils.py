import tiktoken

def get_token_count(prompt: str, model: str = "gpt-4o") -> int:
    """获取与 `get_token_count` 对应的数据或状态。"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

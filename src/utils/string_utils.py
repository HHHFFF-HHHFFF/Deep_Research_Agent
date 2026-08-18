import hashlib
import uuid
from datetime import datetime, timezone


def hash_text_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


def extract_boxed_content(text: str) -> str:
    """提取与 `extract_boxed_content` 对应的数据或状态。"""
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # 说明相关实现细节。
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"


def dedent(text: str) -> str:
    """实现 `dedent` 的业务逻辑。"""
    clean = "\n".join(line.strip() for line in text.splitlines())
    return clean


def generate_unique_id(prefix: str = "session") -> str:
    """生成与 `generate_unique_id` 对应的数据或状态。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"

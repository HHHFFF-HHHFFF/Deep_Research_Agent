from mmengine.registry import Registry

MEMORY_SYSTEM = Registry("memory_system", locations=["src.memory"])
TOOL = Registry("tool", locations=["src.tool"])
ENVIRONMENT = Registry("environment", locations=["src.environment"])
AGENT = Registry("agent", locations=["src.agent"])
PROMPT = Registry("prompt", locations=["src.prompt"])
SKILL = Registry("skill", locations=["src.skill"])

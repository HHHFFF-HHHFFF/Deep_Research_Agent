deep_researcher_tool = {
    "base_dir": "workdir/tool/deep_researcher",
    "model_name": "qwen/qwen3-max",
    # 使用项目自己的网页检索工具，避免绑定某个厂商的内置搜索插件。
    "use_llm_search": False,
    "search_llm_models": [],
    "require_grad": False,
}

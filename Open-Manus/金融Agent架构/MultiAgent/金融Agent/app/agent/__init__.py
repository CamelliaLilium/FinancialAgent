from importlib import import_module

# Lazy exports to avoid importing every optional agent at package import time.
_LAZY_EXPORTS = {
    "BaseAgent": ("app.agent.base", "BaseAgent"),
    "ReActAgent": ("app.agent.react", "ReActAgent"),
    "ToolCallAgent": ("app.agent.toolcall", "ToolCallAgent"),
    "PlanningAgent": ("app.agent.planning", "PlanningAgent"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'app.agent' has no attribute '{name}'")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

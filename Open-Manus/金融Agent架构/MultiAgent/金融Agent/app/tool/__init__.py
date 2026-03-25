from importlib import import_module

# Keep package imports lightweight by lazily loading tool modules on demand.
_LAZY_EXPORTS = {
    "BaseTool": ("app.tool.base", "BaseTool"),
    "ToolCollection": ("app.tool.tool_collection", "ToolCollection"),
    "Terminate": ("app.tool.terminate", "Terminate"),
    "WorkflowStateTool": ("app.tool.workflow_state", "WorkflowStateTool"),
    "PlanningTool": ("app.tool.workflow_state", "PlanningTool"),
    "PythonExecute": ("app.tool.python_execute", "PythonExecute"),
    "StrReplaceEditor": ("app.tool.str_replace_editor", "StrReplaceEditor"),
    "OcrExtract": ("app.tool.ocr", "OcrExtract"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'app.tool' has no attribute '{name}'")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

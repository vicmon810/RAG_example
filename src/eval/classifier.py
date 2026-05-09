def classify_failure(success: bool, stderr: str, generated_code: str) -> str:
    if success:
        return "success"

    if not generated_code.strip():
        return "extraction_error"

    if "SyntaxError" in stderr:
        return "syntax_error"

    if "ModuleNotFoundError" in stderr:
        return "missing_dependency"

    if "NameError" in stderr:
        return "name_error"

    if "KeyError" in stderr:
        return "key_error"

    if "TypeError" in stderr:
        return "type_error"

    if "ValueError" in stderr:
        return "value_error"

    if "TimeoutExpired" in stderr:
        return "timeout"

    if "Traceback" in stderr:
        return "runtime_error"

    return "unknown_error"
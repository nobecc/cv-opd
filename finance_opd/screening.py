"""Separate native runtime readiness from expected student task failures."""


def screening_readiness(outcomes):
    infrastructure, task_failures = [], []
    successful_tools = 0
    for item in outcomes:
        status = item.get("native_status", {})
        successful_tools += status.get("native_tool_successes", 0)
        reason = item.get("error") or status.get("proxy_error")
        if reason:
            infrastructure.append({"id": item["id"], "reason": reason})
            continue
        if not status.get("generated_turns", 0):
            infrastructure.append({"id": item["id"], "reason": "no_verified_student_generation"})
            continue
        task_failure = (status.get("native_status") == "no_finish"
                        or status.get("timed_out") or status.get("context_truncated")
                        or status.get("context_budget_reason"))
        if task_failure:
            task_failures.append({"id": item["id"], "native_status": status})
        elif status.get("exit_code", 0) != 0 or status.get("native_status") == "error":
            infrastructure.append({"id": item["id"], "reason": "unclassified_native_failure"})
    if successful_tools < 1:
        infrastructure.append({"id": None, "reason": "no_successful_native_tool_cycle"})
    return {"passed": bool(outcomes) and not infrastructure,
            "infrastructure_failures": infrastructure, "model_task_failures": task_failures,
            "successful_native_tool_cycles": successful_tools,
            "native_results_unchanged": True}

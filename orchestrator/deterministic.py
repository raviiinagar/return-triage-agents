def check_deterministic_rules(request_data: dict) -> dict:
    """
    Phase 0: Deterministic Initial Validation
    Hard policy gates before agents are invoked.
    """
    order = request_data.get("order", {})
    category = order.get("category", "").lower()
    days_since_delivery = order.get("days_since_delivery", 0)

    # Policy 1: Non-returnable categories
    non_returnable = ["innerwear", "hygiene", "customized", "perishable"]
    if category in non_returnable:
        return {
            "pass": False,
            "reason": f"Category '{category}' is strictly non-returnable."
        }

    # Policy 2: Return windows
    windows = {
        "electronics": 10,
        "clothing": 30,
        "grocery": 2,
        "default": 15
    }
    
    limit = windows.get(category, windows["default"])
    if days_since_delivery > limit:
        return {
            "pass": False,
            "reason": f"Return window expired. {days_since_delivery} days exceeds the {limit}-day limit for '{category}'."
        }

    return {
        "pass": True,
        "reason": "Passed deterministic initial validation."
    }

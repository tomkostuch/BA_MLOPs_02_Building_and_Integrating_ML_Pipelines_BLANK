def format_name(first, last, middle=""):
    """Formats a name as 'Last, First M.' or 'Last, First'."""
    if not first and not last:
        return "" # Added a specific handling for this edge case
    if middle:
        return f"{last}, {first} {middle[0].upper()}."
    return f"{last}, {first}"
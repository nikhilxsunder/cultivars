def _kass_raftery(two_log_bf: float) -> str:
    """Kass and Raftery's verbal scale for ``2 log BF`` in favour of a model."""
    magnitude = abs(two_log_bf)
    if magnitude < 2.0:
        return "not worth more than a bare mention"
    if magnitude < 6.0:
        return "positive"
    if magnitude < 10.0:
        return "strong"
    return "very strong"

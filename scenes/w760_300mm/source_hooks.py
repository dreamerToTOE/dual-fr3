"""Keep scene source substitutions anchored to unique runner code."""


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"Expected exactly one hook: {before[:64]}")
    return source.replace(before, after, 1)

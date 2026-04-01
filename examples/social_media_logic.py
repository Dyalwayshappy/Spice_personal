# Spice Flagship Demo: Social Media Integration

TWITTER_CHAR_LIMIT = 280


def check_post_length(content: str) -> dict:
    """Check whether a post fits within Twitter's 280-character limit.

    Args:
        content: The text content of the post.

    Returns:
        A dict with keys:
            - valid (bool): True if content is within the limit.
            - length (int): Actual character count.
            - limit (int): Platform character limit.
            - remaining (int): Characters remaining (negative if over limit).
    """
    length = len(content)
    remaining = TWITTER_CHAR_LIMIT - length
    return {
        "valid": length <= TWITTER_CHAR_LIMIT,
        "length": length,
        "limit": TWITTER_CHAR_LIMIT,
        "remaining": remaining,
    }

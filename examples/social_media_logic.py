# Spice Flagship Demo: Social Media Integration
# TODO: Implement a logic to check if a tweet length is within the 280 character limit.
# The function check_tweet should return a dictionary with keys: 
# 'is_valid', 'length', and 'remaining'.

def check_tweet(text: str) -> dict:
    MAX_LENGTH = 280
    length = len(text)
    return {
        "is_valid": length <= MAX_LENGTH,
        "length": length,
        "remaining": MAX_LENGTH - length,
    }

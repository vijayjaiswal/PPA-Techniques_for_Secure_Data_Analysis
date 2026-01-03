import hashlib

def sha256_hash(data: str) -> str:
    """
    Generate a SHA-256 hash for the given input string.
    """
    hash_object = hashlib.sha256(data.encode('utf-8'))
    return hash_object.hexdigest()

# Example usage
message = "Sensitive data"
hash_value = sha256_hash(message)
print("SHA-256 Hash:", hash_value)

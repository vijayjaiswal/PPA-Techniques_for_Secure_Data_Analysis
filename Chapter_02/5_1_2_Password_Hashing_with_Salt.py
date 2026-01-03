import hashlib
import os

def hash_password(password: str) -> tuple:
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return salt, pwd_hash

# Example usage
salt, stored_hash = hash_password("MySecurePassword")
print("Salt:", salt)
print("Password Hash:", stored_hash)

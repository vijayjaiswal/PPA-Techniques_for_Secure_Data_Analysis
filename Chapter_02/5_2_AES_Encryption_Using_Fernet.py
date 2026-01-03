from cryptography.fernet import Fernet

# Generate a secret key
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt data
plaintext = b"Confidential analytics data"
ciphertext = cipher.encrypt(plaintext)

# Decrypt data
decrypted_text = cipher.decrypt(ciphertext)

print("Ciphertext:", ciphertext)
print("Decrypted:", decrypted_text)

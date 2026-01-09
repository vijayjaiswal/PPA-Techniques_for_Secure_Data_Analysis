from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Generate RSA private key with 2048-bit key size for balanced security and performance
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Derive the public key from the private key
public_key = private_key.public_key()

# Optional: Serialize keys for storage or transmission purposes
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
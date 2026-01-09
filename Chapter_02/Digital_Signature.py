from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

data = b"Sensitive analytics dataset record"

# Sign data by hashing and encrypting with private key using PSS padding
signature = private_key.sign(
    data,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)
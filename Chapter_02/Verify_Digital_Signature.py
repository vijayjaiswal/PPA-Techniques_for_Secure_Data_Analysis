try:
    # Verify signature by decrypting using public key and matching the hash
    public_key.verify(
        signature,
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("Signature verified successfully: Data is authentic and unaltered.")
except Exception:
    print("Signature verification failed: Data integrity or authenticity compromised.")
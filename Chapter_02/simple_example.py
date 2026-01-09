"""
Simple example demonstrating basic RSA signature usage.
"""

from rsa_signature import RSASignature

def simple_example():
    """Simple demonstration of RSA signatures."""
    print("Simple RSA Signature Example")
    print("-" * 40)
    
    # Create handler and generate keys
    rsa = RSASignature()
    rsa.generate_keys()
    
    # Data to sign
    message = b"Important message to sign"
    print(f"Original message: {message.decode()}")
    
    # Create signature
    signature = rsa.sign_data(message)
    print(f"\nSignature created: {signature[:20].hex()}... (first 20 bytes)")
    print(f"Signature length: {len(signature)} bytes")
    
    # Verify signature
    print("\nVerifying original signature...")
    if rsa.verify_signature(message, signature):
        print("✓ Signature is valid!")
    else:
        print("✗ Signature is invalid!")
    
    # Try with modified message
    print("\nVerifying signature with modified message...")
    modified_message = b"Important message to sign - MODIFIED"
    if not rsa.verify_signature(modified_message, signature):
        print("✓ Correctly rejected modified message!")
    else:
        print("✗ Failed to detect modified message!")
    
    # Show key info
    private_key, public_key = rsa.get_keys()
    print(f"\nKey information:")
    print(f"  Key size: {private_key.key_size} bits")
    print(f"  Public exponent: {private_key.public_key().public_numbers().e}")

if __name__ == "__main__":
    simple_example()
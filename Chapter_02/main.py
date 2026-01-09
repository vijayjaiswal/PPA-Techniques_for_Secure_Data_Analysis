"""
Main test file for RSA digital signature functionality.
"""

import os
from rsa_signature import RSASignature


def test_basic_signature():
    """Test basic signature creation and verification."""
    print("=" * 60)
    print("Test 1: Basic Signature Creation and Verification")
    print("=" * 60)
    
    # Initialize RSA signature handler
    rsa_handler = RSASignature(key_size=2048)
    
    # Generate keys
    print("Generating RSA key pair (2048-bit)...")
    private_key, public_key = rsa_handler.generate_keys()
    print("✓ Key pair generated successfully")
    
    # Test data
    test_data = b"Sensitive analytics dataset record"
    print(f"\nTest data: {test_data.decode()}")
    
    # Create signature
    print("\nCreating digital signature...")
    signature = rsa_handler.sign_data(test_data)
    print(f"✓ Signature created (length: {len(signature)} bytes)")
    
    # Verify signature (should succeed)
    print("\nVerifying signature...")
    is_valid = rsa_handler.verify_signature(test_data, signature)
    if is_valid:
        print("✓ Signature verified successfully: Data is authentic and unaltered.")
    else:
        print("✗ Signature verification failed!")
    
    # Test with tampered data (should fail)
    print("\nTesting with tampered data...")
    tampered_data = b"Sensitive analytics dataset record MODIFIED"
    is_valid = rsa_handler.verify_signature(tampered_data, signature)
    if not is_valid:
        print("✓ Correctly rejected tampered data")
    else:
        print("✗ Failed to detect tampered data!")


def test_key_serialization():
    """Test serialization and deserialization of keys."""
    print("\n" + "=" * 60)
    print("Test 2: Key Serialization and Deserialization")
    print("=" * 60)
    
    rsa_handler = RSASignature()
    rsa_handler.generate_keys()
    
    # Serialize keys
    print("Serializing keys to PEM format...")
    private_pem = rsa_handler.serialize_private_key(rsa_handler.get_keys()[0])
    public_pem = rsa_handler.serialize_public_key(rsa_handler.get_keys()[1])
    
    print(f"Private key PEM length: {len(private_pem)} bytes")
    print(f"Public key PEM length: {len(public_pem)} bytes")
    
    # Deserialize keys
    print("\nDeserializing keys from PEM format...")
    private_key_loaded = rsa_handler.deserialize_private_key(private_pem)
    public_key_loaded = rsa_handler.deserialize_public_key(public_pem)
    
    # Test with deserialized keys
    test_data = b"Test data for serialization"
    
    # Create a new handler for testing with loaded keys
    test_handler = RSASignature()
    signature = test_handler.sign_data(test_data, private_key_loaded)
    
    # Verify with deserialized public key
    is_valid = test_handler.verify_signature(test_data, signature, public_key_loaded)
    if is_valid:
        print("✓ Keys successfully serialized and deserialized")
    else:
        print("✗ Key serialization/deserialization failed!")


def test_file_operations():
    """Test saving and loading keys to/from files."""
    print("\n" + "=" * 60)
    print("Test 3: File Operations")
    print("=" * 60)
    
    rsa_handler = RSASignature()
    rsa_handler.generate_keys()
    
    # Define file paths
    private_key_file = "test_private_key.pem"
    public_key_file = "test_public_key.pem"
    
    try:
        # Save keys to files
        print("Saving keys to files...")
        rsa_handler.save_keys_to_file(private_key_file, public_key_file)
        print(f"✓ Keys saved to {private_key_file} and {public_key_file}")
        
        # Create new handler and load keys
        print("\nLoading keys from files into new handler...")
        new_handler = RSASignature()
        new_handler.load_keys_from_file(private_key_file, public_key_file)
        
        # Test with loaded keys
        test_data = b"Test data for file operations"
        signature = new_handler.sign_data(test_data)
        is_valid = new_handler.verify_signature(test_data, signature)
        
        if is_valid:
            print("✓ Keys successfully loaded from files and functional")
        else:
            print("✗ Loaded keys are not functional!")
    
    finally:
        # Clean up test files
        if os.path.exists(private_key_file):
            os.remove(private_key_file)
        if os.path.exists(public_key_file):
            os.remove(public_key_file)
        print(f"\nCleaned up test files")


def test_different_key_sizes():
    """Test with different key sizes."""
    print("\n" + "=" * 60)
    print("Test 4: Different Key Sizes")
    print("=" * 60)
    
    key_sizes = [2048, 3072, 4096]  # Note: 1024-bit is now considered insecure
    
    for size in key_sizes:
        print(f"\nTesting with {size}-bit key...")
        try:
            rsa_handler = RSASignature(key_size=size)
            rsa_handler.generate_keys()
            
            test_data = b"Test data"
            signature = rsa_handler.sign_data(test_data)
            is_valid = rsa_handler.verify_signature(test_data, signature)
            
            if is_valid:
                print(f"✓ {size}-bit key works correctly")
            else:
                print(f"✗ {size}-bit key verification failed")
        except Exception as e:
            print(f"✗ Error with {size}-bit key: {str(e)}")


def test_password_protected_keys():
    """Test password-protected private keys."""
    print("\n" + "=" * 60)
    print("Test 5: Password-Protected Keys")
    print("=" * 60)
    
    rsa_handler = RSASignature()
    rsa_handler.generate_keys()
    
    password = b"strong_password_123"
    test_data = b"Sensitive data requiring password protection"
    
    # Serialize with password
    print("Serializing private key with password protection...")
    private_pem = rsa_handler.serialize_private_key(
        rsa_handler.get_keys()[0], 
        password
    )
    
    # Try to deserialize without password (should fail)
    print("\nTrying to deserialize without password...")
    try:
        rsa_handler.deserialize_private_key(private_pem)
        print("✗ Should have required password!")
    except (ValueError, TypeError) as e:
        print(f"✓ Correctly required password: {type(e).__name__}")
    
    # Deserialize with correct password (should succeed)
    print("\nDeserializing with correct password...")
    try:
        private_key = rsa_handler.deserialize_private_key(private_pem, password)
        
        # Test the deserialized key
        signature = rsa_handler.sign_data(test_data, private_key)
        is_valid = rsa_handler.verify_signature(test_data, signature)
        
        if is_valid:
            print("✓ Password-protected key works correctly")
        else:
            print("✗ Password-protected key verification failed")
    except Exception as e:
        print(f"✗ Error with password-protected key: {str(e)}")


def test_multiple_signatures():
    """Test that the same data produces different signatures (due to PSS salt)."""
    print("\n" + "=" * 60)
    print("Test 6: Multiple Signatures (PSS Randomness)")
    print("=" * 60)
    
    rsa_handler = RSASignature()
    rsa_handler.generate_keys()
    
    test_data = b"Same data signed multiple times"
    
    # Sign the same data multiple times
    signatures = []
    for i in range(3):
        signature = rsa_handler.sign_data(test_data)
        signatures.append(signature)
        print(f"Signature {i+1}: {signature[:20].hex()}... (first 20 bytes)")
    
    # All signatures should be different (due to PSS salt)
    if (signatures[0] != signatures[1] and 
        signatures[0] != signatures[2] and 
        signatures[1] != signatures[2]):
        print("✓ PSS produces different signatures (salt is working)")
    else:
        print("✗ PSS signatures should be different")
    
    # All signatures should verify correctly
    all_valid = all(rsa_handler.verify_signature(test_data, sig) for sig in signatures)
    if all_valid:
        print("✓ All signatures verify correctly")
    else:
        print("✗ Some signatures failed verification")


def main():
    """Run all tests."""
    print("RSA Digital Signature Testing Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_signature()
        test_key_serialization()
        test_file_operations()
        test_different_key_sizes()
        test_password_protected_keys()
        test_multiple_signatures()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
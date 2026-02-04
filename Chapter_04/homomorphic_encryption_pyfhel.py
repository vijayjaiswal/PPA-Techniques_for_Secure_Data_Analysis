import numpy as np
from Pyfhel import Pyfhel

def run_homomorphic_demo():
    print("--- 1. Context Setup ---")
    HE = Pyfhel()
    
    # contextGen: Sets up the cryptographic parameters.
    # 'bfv' is a scheme supported by HElib/SEAL for exact integer arithmetic.
    # n=4096: The polynomial modulus degree (higher = more security/space).
    # t_bits=20: The number of bits for the plaintext modulus.
    HE.contextGen(scheme='bfv', n=4096, t_bits=20)
    
    # Generate the Public and Private keys
    HE.keyGen()
    
    print(f"Context generated. Scheme: {HE.scheme}")

    print("\n--- 2. Encryption ---")
    integer1 = np.array([15], dtype=np.int64)
    integer2 = np.array([5], dtype=np.int64)
    
    # Encrypt the integers
    # The 'ciphertext' object (ctxt1) holds the encrypted data.
    # You cannot read the value inside ctxt1 without the private key.
    ctxt1 = HE.encryptInt(integer1)
    ctxt2 = HE.encryptInt(integer2)
    
    print(f"Plaintext inputs: {integer1}, {integer2}")
    print(f"Ciphertext objects created: {type(ctxt1)}")

    print("\n--- 3. Encrypted Operations ---")
    # Addition
    # The operation happens directly on the ciphertexts.
    # The CPU is adding polynomials, not the numbers 15 and 5.
    ctxt_sum = ctxt1 + ctxt2
    
    # Multiplication
    ctxt_prod = ctxt1 * ctxt2
    
    # In-place modification (Efficiency tip)
    # ctxt1 += ctxt2  <-- This would modify ctxt1 directly to save memory
    
    print("Operations completed on encrypted data.")

    print("\n--- 4. Decryption ---")
    # Use the private key (stored inside HE object) to decrypt
    res_sum = HE.decryptInt(ctxt_sum)
    res_prod = HE.decryptInt(ctxt_prod)
    
    print(f"Decrypted Sum: {res_sum}  (Expected: {integer1 + integer2})")
    print(f"Decrypted Product: {res_prod} (Expected: {integer1 * integer2})")

if __name__ == "__main__":
    run_homomorphic_demo()
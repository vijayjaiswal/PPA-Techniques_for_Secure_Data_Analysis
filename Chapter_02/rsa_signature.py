"""
RSA Digital Signature Module
Provides functionality for generating RSA keys, creating signatures, and verifying them.
"""

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.exceptions import InvalidSignature
from typing import Tuple, Optional, Union
import os


class RSASignature:
    """Class for handling RSA digital signatures."""
    
    def __init__(self, key_size: int = 2048, public_exponent: int = 65537):
        """
        Initialize RSA Signature handler.
        
        Args:
            key_size: RSA key size in bits (default: 2048)
            public_exponent: Public exponent (default: 65537)
        """
        self.key_size = key_size
        self.public_exponent = public_exponent
        self._private_key = None
        self._public_key = None
    
    def generate_keys(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Generate RSA key pair.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        self._private_key = rsa.generate_private_key(
            public_exponent=self.public_exponent,
            key_size=self.key_size
        )
        self._public_key = self._private_key.public_key()
        return self._private_key, self._public_key
    
    def sign_data(self, data: bytes, private_key: Optional[rsa.RSAPrivateKey] = None) -> bytes:
        """
        Create a digital signature for the given data.
        
        Args:
            data: Data to sign (bytes)
            private_key: Private key to use (uses instance key if None)
            
        Returns:
            Digital signature (bytes)
        """
        if private_key is None:
            if self._private_key is None:
                raise ValueError("No private key available. Generate keys first or provide a private key.")
            private_key = self._private_key
        
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, 
                        public_key: Optional[rsa.RSAPublicKey] = None) -> bool:
        """
        Verify a digital signature.
        
        Args:
            data: Original data (bytes)
            signature: Digital signature to verify (bytes)
            public_key: Public key to use for verification
            
        Returns:
            True if signature is valid, False otherwise
        """
        if public_key is None:
            if self._public_key is None:
                raise ValueError("No public key available. Provide a public key.")
            public_key = self._public_key
        
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            raise ValueError(f"Verification failed: {str(e)}")
    
    def get_keys(self) -> Tuple[Optional[rsa.RSAPrivateKey], Optional[rsa.RSAPublicKey]]:
        """
        Get the current keys.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        return self._private_key, self._public_key
    
    @staticmethod
    def serialize_private_key(private_key: rsa.RSAPrivateKey, 
                             password: Optional[bytes] = None) -> bytes:
        """
        Serialize private key to PEM format.
        
        Args:
            private_key: RSA private key
            password: Optional password for encryption (bytes)
            
        Returns:
            Serialized private key (bytes)
        """
        if password:
            encryption_algorithm = serialization.BestAvailableEncryption(password)
        else:
            encryption_algorithm = serialization.NoEncryption()
        
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
    
    @staticmethod
    def serialize_public_key(public_key: rsa.RSAPublicKey) -> bytes:
        """
        Serialize public key to PEM format.
        
        Args:
            public_key: RSA public key
            
        Returns:
            Serialized public key (bytes)
        """
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    @staticmethod
    def deserialize_private_key(pem_data: bytes, 
                               password: Optional[bytes] = None) -> rsa.RSAPrivateKey:
        """
        Deserialize private key from PEM format.
        
        Args:
            pem_data: PEM-encoded private key (bytes)
            password: Optional password if key is encrypted
            
        Returns:
            RSA private key
        """
        return serialization.load_pem_private_key(pem_data, password=password)
    
    @staticmethod
    def deserialize_public_key(pem_data: bytes) -> rsa.RSAPublicKey:
        """
        Deserialize public key from PEM format.
        
        Args:
            pem_data: PEM-encoded public key (bytes)
            
        Returns:
            RSA public key
        """
        return serialization.load_pem_public_key(pem_data)
    
    def save_keys_to_file(self, private_key_path: str, public_key_path: str, 
                         password: Optional[bytes] = None):
        """
        Save keys to files.
        
        Args:
            private_key_path: Path to save private key
            public_key_path: Path to save public key
            password: Optional password for private key encryption
        """
        if self._private_key is None or self._public_key is None:
            raise ValueError("No keys available. Generate keys first.")
        
        # Save private key
        with open(private_key_path, 'wb') as f:
            f.write(self.serialize_private_key(self._private_key, password))
        
        # Save public key
        with open(public_key_path, 'wb') as f:
            f.write(self.serialize_public_key(self._public_key))
    
    def load_keys_from_file(self, private_key_path: str, 
                           public_key_path: Optional[str] = None,
                           password: Optional[bytes] = None):
        """
        Load keys from files.
        
        Args:
            private_key_path: Path to private key file
            public_key_path: Path to public key file (optional, can derive from private)
            password: Optional password for private key decryption
        """
        # Load private key
        with open(private_key_path, 'rb') as f:
            self._private_key = self.deserialize_private_key(f.read(), password)
        
        # Load or derive public key
        if public_key_path:
            with open(public_key_path, 'rb') as f:
                self._public_key = self.deserialize_public_key(f.read())
        else:
            self._public_key = self._private_key.public_key()
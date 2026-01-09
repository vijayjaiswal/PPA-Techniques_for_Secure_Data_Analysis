import json
import base64
import pickle
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import hashlib
from typing import Dict, Any, Tuple, Union
import datetime


class SignedDataTransmitter:
    """
    Demonstrates transmission/storage of data with digital signature.
    The signature serves as a verifiable seal binding the signer to the data.
    """

    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
        print(f"Generated {self.key_size}-bit RSA key pair")

    """
    Hashing the Data
    Encrypting the Hash with the Private Key
    Base64 Encoding the Signature
    """
    def create_signature(self, data: bytes) -> bytes:
        """Create digital signature for data."""
        # Hash data using SHA-256
        hash_digest = hashlib.sha256(data).digest()

        # Sign data by hashing and encrypting with private key using PSS padding
        signature = self.private_key.sign(
            hash_digest,
            padding.PSS(
                mgf = padding.MGF1(hashes.SHA256()),
                salt_length = padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def create_signed_package_json(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a signed package in JSON format for transmission.

        Args:
            data_dict: Data to sign and transmit

        Returns:
            Dictionary containing data and signature
        """
        print("\n" + "=" * 70)
        print("CREATING SIGNED PACKAGE FOR TRANSMISSION (JSON)")
        print("=" * 70)

        # Convert data to bytes for signing
        data_bytes = json.dumps(data_dict, sort_keys=True).encode('utf-8')

        # Create signature
        signature = self.create_signature(data_bytes)

        # Create transmission package
        package = {
            "metadata": {
                "version": "1.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "algorithm": "RSA-SHA256-PKCS1v15",
                "key_size": self.key_size
            },
            "data": data_dict,
            "signature": base64.b64encode(signature).decode('utf-8')
        }

        print(f"Original data: {json.dumps(data_dict, indent=2)}")
        print(f"\nSignature (base64): {package['signature'][:80]}...")
        print(f"Package size: {len(json.dumps(package).encode('utf-8'))} bytes")

        return package

    def transmit_json_package(self, package: Dict[str, Any]) -> str:
        """
        Simulate transmission by converting to JSON string.

        Args:
            package: Signed package

        Returns:
            JSON string ready for transmission
        """
        return json.dumps(package)

    def receive_and_verify_json(self, json_string: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Receive and verify transmitted JSON package.

        Args:
            json_string: Received JSON string

        Returns:
            Tuple of (is_valid, data)
        """
        print("\n" + "=" * 70)
        print("RECEIVING AND VERIFYING TRANSMITTED PACKAGE")
        print("=" * 70)

        # Parse received package
        package = json.loads(json_string)

        print(f"Received package metadata:")
        print(f"  Timestamp: {package['metadata']['timestamp']}")
        print(f"  Algorithm: {package['metadata']['algorithm']}")

        # Extract data and signature
        data_dict = package['data']
        signature = base64.b64decode(package['signature'])

        # Convert data back to bytes for verification
        data_bytes = json.dumps(data_dict, sort_keys=True).encode('utf-8')

        # Hash the received data
        received_hash = hashlib.sha256(data_bytes).digest()

        # Verify signature
        try:
            self.public_key.verify(
                signature,
                received_hash,
                #padding.MGF1(hashes.SHA256()),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            print("✓ SIGNATURE VERIFIED: Data is authentic and intact")
            print(f"  Data: {json.dumps(data_dict, indent=2)}")
            return True, data_dict
        except Exception as e:
            print(f"✗ SIGNATURE VERIFICATION FAILED: {e}")
            return False, data_dict

    def create_signed_binary_package(self, data: bytes, metadata: Dict[str, Any] = None) -> bytes:
        """
        Create binary package for more efficient transmission.

        Args:
            data: Binary data to sign
            metadata: Optional metadata

        Returns:
            Binary package
        """
        print("\n" + "=" * 70)
        print("CREATING BINARY SIGNED PACKAGE")
        print("=" * 70)

        # Create signature
        signature = self.create_signature(data)

        # Create metadata
        if metadata is None:
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "data_size": len(data),
                "signature_size": len(signature)
            }

        # Create binary package structure
        package = {
            'metadata': metadata,
            'data': data,
            'signature': signature
        }

        # Serialize to binary
        binary_package = pickle.dumps(package)

        print(f"Original data size: {len(data)} bytes")
        print(f"Signature size: {len(signature)} bytes")
        print(f"Total package size: {len(binary_package)} bytes")
        print(f"Overhead: {len(binary_package) - len(data)} bytes")

        return binary_package

    def verify_binary_package(self, binary_package: bytes) -> Tuple[bool, bytes]:
        """
        Verify binary package.

        Args:
            binary_package: Received binary package

        Returns:
            Tuple of (is_valid, data)
        """
        # Deserialize package
        package = pickle.loads(binary_package)

        # Extract components
        data = package['data']
        signature = package['signature']

        # Hash the data
        data_hash = hashlib.sha256(data).digest()

        # Verify signature
        try:
            self.public_key.verify(
                signature,
                data_hash,
                padding.MGF1(hashes.SHA256()),
                hashes.SHA256()
            )
            return True, data
        except Exception:
            return False, data


class SecureMessageTransmitter:
    """
    Example: Secure message transmission with signatures.
    """

    def __init__(self, sender_name: str):
        self.sender_name = sender_name
        self.key_pair = SignedDataTransmitter()

    def send_message(self, recipient: str, message: str, urgent: bool = False) -> str:
        """
        Create and transmit a signed message.

        Args:
            recipient: Recipient name
            message: Message content
            urgent: Whether message is urgent

        Returns:
            JSON string ready for transmission
        """
        # Create message data
        message_data = {
            "sender": self.sender_name,
            "recipient": recipient,
            "message": message,
            "urgent": urgent,
            "message_id": f"msg_{datetime.datetime.now().timestamp()}"
        }

        # Create signed package
        package = self.key_pair.create_signed_package_json(message_data)

        # Simulate transmission
        transmitted = self.key_pair.transmit_json_package(package)

        print(f"\n📤 TRANSMISSION FROM {self.sender_name} TO {recipient}")
        print(f"   Message: '{message[:50]}...'" if len(message) > 50 else f"   Message: '{message}'")
        print(f"   Transmission ready: {len(transmitted)} bytes")

        return transmitted

    def receive_message(self, json_string: str) -> None:
        """
        Receive and verify a message.

        Args:
            json_string: Received JSON string
        """
        is_valid, data = self.key_pair.receive_and_verify_json(json_string)

        if is_valid:
            print(f"\n📥 MESSAGE RECEIVED:")
            print(f"   From: {data['sender']}")
            print(f"   To: {data['recipient']}")
            print(f"   Message: {data['message']}")
            if data.get('urgent'):
                print(f"   ⚠️  URGENT MESSAGE")
            print(f"   ✓ VERIFIED: Authentic from {data['sender']}")
        else:
            print(f"\n❌ MESSAGE REJECTED: Signature verification failed")


def demonstrate_transmission_scenarios():
    """Demonstrate different transmission scenarios."""
    print("TRANSMISSION OF DATA AND SIGNATURE DEMONSTRATION")
    print("=" * 70)
    print("\nPrinciple: Both data and signature are transmitted together.")
    print("The signature serves as a verifiable seal binding the signer to the data.\n")

    # Scenario 1: JSON transmission
    print("SCENARIO 1: JSON Transmission")
    print("-" * 40)

    transmitter = SignedDataTransmitter()

    # Create financial transaction data
    transaction = {
        "transaction_id": "TX123456789",
        "from_account": "ACC001",
        "to_account": "ACC002",
        "amount": 10000.00,
        "currency": "USD",
        "description": "Invoice payment",
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Create and transmit signed package
    package = transmitter.create_signed_package_json(transaction)
    json_string = transmitter.transmit_json_package(package)

    # Simulate transmission (could be HTTP, messaging queue, file, etc.)
    print(f"\n📡 Transmitting over network...")
    print(f"   Format: JSON")
    print(f"   Size: {len(json_string)} characters")

    # Receive and verify
    is_valid, received_data = transmitter.receive_and_verify_json(json_string)

    # Scenario 2: Message exchange
    print("\n\nSCENARIO 2: Secure Messaging")
    print("-" * 40)

    alice = SecureMessageTransmitter("Alice")
    bob = SecureMessageTransmitter("Bob")

    # Alice sends message to Bob
    message = "Meeting scheduled for tomorrow at 10 AM. Please bring the quarterly reports."
    transmitted = alice.send_message("Bob", message, urgent=True)

    # Bob receives (simulate transmission)
    print("\n📡 Transmission in progress...")

    # Bob receives and verifies
    bob.receive_message(transmitted)

    # Scenario 3: Tampered transmission attempt
    print("\n\nSCENARIO 3: Tampered Transmission Detection")
    print("-" * 40)

    # Create legitimate package
    legit_package = transmitter.create_signed_package_json({"status": "approved"})
    legit_json = json.dumps(legit_package)

    # Tamper with the data
    tampered_dict = json.loads(legit_json)
    tampered_dict['data']['status'] = "rejected"  # Change data
    tampered_json = json.dumps(tampered_dict)

    print(f"\n⚠️  ATTEMPT: Malicious actor intercepts and modifies data")
    print(f"   Original status: 'approved'")
    print(f"   Modified status: 'rejected'")

    # Try to verify tampered package
    is_valid, data = transmitter.receive_and_verify_json(tampered_json)

    if not is_valid:
        print(f"\n✅ SECURITY: Tampering detected and rejected!")

    # Scenario 4: Binary file storage
    print("\n\nSCENARIO 4: Secure File Storage")
    print("-" * 40)

    # Create sensitive document
    document_content = b"CONFIDENTIAL: Project Athena Specifications\n" \
                       b"Budget: $2,500,000\n" \
                       b"Timeline: 12 months\n" \
                       b"Team: 15 members\n" \
                       b"Status: Approved by board"

    # Create binary package
    binary_package = transmitter.create_signed_binary_package(
        document_content,
        metadata={
            "filename": "project_athena.txt",
            "author": "CTO Office",
            "classification": "CONFIDENTIAL"
        }
    )

    # Save to file (simulating storage)
    with open("secured_document.bin", "wb") as f:
        f.write(binary_package)

    print(f"\n💾 File saved: secured_document.bin")
    print(f"   Can be stored or transmitted securely")

    # Later: Verify when reading
    with open("secured_document.bin", "rb") as f:
        loaded_package = f.read()

    is_valid, retrieved_data = transmitter.verify_binary_package(loaded_package)

    if is_valid:
        print(f"📄 File retrieved and verified:")
        print(f"   Content: {retrieved_data[:50].decode()}...")
        print(f"   ✓ Integrity confirmed")

    # Cleanup
    import os
    if os.path.exists("secured_document.bin"):
        os.remove("secured_document.bin")


# Minimal essential example
def minimal_transmission_example():
    """Minimal example showing the core concept."""
    print("\n" + "=" * 70)
    print("MINIMAL ESSENTIAL EXAMPLE")
    print("=" * 70)

    # Generate keys
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Data to transmit
    data = b"Critical system configuration"

    # Create signature
    hash_digest = hashlib.sha256(data).digest()
    signature = private_key.sign(
        hash_digest,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    # Package for transmission (data + signature)
    transmission_package = {
        'data': data,
        'signature': signature
    }

    print(f"🔐 Creating transmission package:")
    print(f"   Data: {data.decode()}")
    print(f"   Signature size: {len(signature)} bytes")

    # Simulate transmission
    print(f"\n📡 Transmitting package...")
    print(f"   Both data and signature are sent together")

    # Receive and verify
    print(f"\n📥 Receiving package...")

    # Extract from package
    received_data = transmission_package['data']
    received_signature = transmission_package['signature']

    # Verify
    received_hash = hashlib.sha256(received_data).digest()

    try:
        public_key.verify(
            received_signature,
            received_hash,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        print(f"   ✓ Package verified: Data is authentic")
        print(f"   The signature binds the signer to this specific data")
    except:
        print(f"   ✗ Verification failed")


if __name__ == "__main__":
    # Run full demonstration
    demonstrate_transmission_scenarios()

    # Run minimal example
    #minimal_transmission_example()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Data and signature travel together as a package")
    print("2. Signature serves as a 'seal' authenticating the data")
    print("3. Anyone with the public key can verify but not forge")
    print("4. Tampering is immediately detectable")
    print("5. Enables trust in distributed/networked systems")
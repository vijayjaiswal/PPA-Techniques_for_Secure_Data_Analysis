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

    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
        print(f"Generated {self.key_size}-bit RSA key pair")

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
        return json.dumps(package)

    def receive_and_verify_json(self, json_string: str) -> Tuple[bool, Dict[str, Any]]:

        print("\n" +"RECEIVING AND VERIFYING TRANSMITTED PACKAGE")
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
                padding.PSS(
                mgf = padding.MGF1(hashes.SHA256()),
                salt_length = padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            print("✓ SIGNATURE VERIFIED: Data is authentic and intact")
            print(f"  Data: {json.dumps(data_dict, indent=2)}")
            return True, data_dict
        except Exception as e:
            print(f"✗ SIGNATURE VERIFICATION FAILED: {e}")
            return False, data_dict

class SecureMessageTransmitter:
    def __init__(self, sender_name: str):
        self.sender_name = sender_name
        self.key_pair = SignedDataTransmitter()

    def send_message(self, recipient: str, message: str, urgent: bool = False) -> str:
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
    print("TRANSMISSION OF DATA AND SIGNATURE DEMONSTRATION")
    print("=" * 70)
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

   

    

   

if __name__ == "__main__":
    # Run full demonstration
    demonstrate_transmission_scenarios()

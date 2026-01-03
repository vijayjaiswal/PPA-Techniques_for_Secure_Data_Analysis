import boto3
import base64
import json

kms = boto3.client('kms', region_name='us-east-2')

# Sample Data
plaintext = "My secret data"
print("Plaintext:", plaintext)

# Encrypt
response = kms.encrypt(KeyId='530aed6d-9f33-4d7d-b258-eeb04a661242', Plaintext=plaintext.encode())
ciphertext = response['CiphertextBlob']
print("Ciphertext:", base64.b64encode(ciphertext).decode())

# Decrypt
response = kms.decrypt(CiphertextBlob=ciphertext)
decrypted = response['Plaintext'].decode()
print("Decrypted:", decrypted)

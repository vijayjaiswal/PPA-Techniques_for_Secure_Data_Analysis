# 1. Check OpenSSL Version
openssl version

# 2. Hashing (SHA-256) – Data Integrity
echo "Sensitive data" | openssl dgst -sha256

# 3. Generate Random Data
openssl rand -hex 32

# 4. Symmetric Encryption (AES-256-CBC)
# Encrypt a File
openssl enc -aes-256-cbc -salt -in data.txt -out data.enc

# Decrypt a File
openssl enc -aes-256-cbc -d -in data.enc -out data_dec.txt

# 5. Symmetric Encryption (AES-256-GCM – Recommended)
openssl enc -aes-256-gcm -salt -in data.txt -out data.enc

# 6. Generate RSA Key Pair
# Private Key (2048-bit)
openssl genpkey -algorithm RSA -out private_key.pem -pkeyopt rsa_keygen_bits:2048

# Public Key
openssl rsa -pubout -in private_key.pem -out public_key.pem

# 7. Digital Signatures (RSA + SHA-256)
# Sign a File
openssl dgst -sha256 -sign private_key.pem -out signature.bin data.txt

# Verify a Signature
openssl dgst -sha256 -verify public_key.pem -signature signature.bin data.txt

# 8. Generate a Self-Signed X.509 Certificate
openssl req -x509 -new -key private_key.pem -out certificate.pem -days 365 -config "D:\installed\OpenSSL-Win64\bin\cnf\openssl-vms.cnf"

# 9. View Certificate Details
openssl x509 -in certificate.pem -text -noout

# 10. Encrypt a Symmetric Key with RSA
openssl rsautl -encrypt -pubin -inkey public_key.pem -in aes.key -out aes.key.enc

# 11. Verify File Integrity
openssl dgst -sha256 data.txt



"""
Chapter 10 - Custom Audits with Python:
Writing Scripts for Security Compliance Verification

This module demonstrates:
1. File Permission Auditing - Check for overly permissive files
2. Log File Anomaly Detection - Parse logs for suspicious patterns
3. S3 Encryption Status Validation - Verify AES-256 on buckets (simulated)
4. Sensitive Data Scanner - Detect SSNs, credit cards, API keys in logs
5. Consolidated Audit Report Generation
"""

import os
import re
import stat
import json
import hashlib
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1: File Permission Auditing
# =============================================================================

# --- Patterns for sensitive files that should be locked down ---
SENSITIVE_PATTERNS = [
    r'\.pem$', r'\.key$', r'\.env$', r'\.cfg$', r'\.ini$',
    r'\.conf$', r'password', r'secret', r'credential', r'\.p12$',
]

PERMISSION_POLICY = {
    "private_keys":   {"max_mode": 0o600, "patterns": [r'\.pem$', r'\.key$', r'\.p12$']},
    "config_files":   {"max_mode": 0o644, "patterns": [r'\.env$', r'\.cfg$', r'\.ini$', r'\.conf$']},
    "secret_files":   {"max_mode": 0o600, "patterns": [r'password', r'secret', r'credential']},
}


def check_file_permission(filepath, max_mode):
    """Checks if a file's permissions exceed the allowed maximum.

    Args:
        filepath: Absolute path to the file.
        max_mode: Maximum allowed permission mode (e.g., 0o600).

    Returns:
        dict with finding details, or None if compliant.
    """
    try:
        file_stat = os.stat(filepath)
        current_mode = stat.S_IMODE(file_stat.st_mode)

        # Check if any extra permission bits are set beyond the policy
        violation_bits = current_mode & ~max_mode
        if violation_bits:
            return {
                "file": filepath,
                "current_permissions": oct(current_mode),
                "max_allowed": oct(max_mode),
                "violation_bits": oct(violation_bits),
                "severity": "CRITICAL" if (violation_bits & 0o007) else "WARNING",
                "recommendation": f"Run: chmod {oct(max_mode)} {filepath}",
            }
    except (OSError, PermissionError) as e:
        return {
            "file": filepath,
            "error": str(e),
            "severity": "ERROR",
            "recommendation": "Unable to stat file - check access.",
        }
    return None


def audit_file_permissions(scan_dir):
    """Scans a directory tree for files with overly permissive access.

    Args:
        scan_dir: Root directory to scan.

    Returns:
        list of finding dicts for non-compliant files.
    """
    print("=" * 70)
    print("SECTION 1: File Permission Audit")
    print("=" * 70)
    print(f"\n  Scanning directory: {scan_dir}")

    findings = []
    files_scanned = 0

    for root, dirs, files in os.walk(scan_dir):
        # Skip hidden/virtual-env directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   ('node_modules', '__pycache__', 'venv', '.venv')]
        for fname in files:
            filepath = os.path.join(root, fname)
            files_scanned += 1

            for category, policy in PERMISSION_POLICY.items():
                for pattern in policy["patterns"]:
                    if re.search(pattern, fname, re.IGNORECASE):
                        finding = check_file_permission(filepath, policy["max_mode"])
                        if finding:
                            finding["category"] = category
                            findings.append(finding)
                        break

    # Summary
    critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    warning = sum(1 for f in findings if f.get("severity") == "WARNING")

    print(f"  Files scanned : {files_scanned}")
    print(f"  Violations    : {len(findings)} "
          f"({critical} critical, {warning} warnings)\n")

    for f in findings:
        icon = "!!" if f["severity"] == "CRITICAL" else "**"
        print(f"  [{icon}] {f['severity']}: {f.get('file', 'N/A')}")
        if "current_permissions" in f:
            print(f"       Current: {f['current_permissions']}  "
                  f"Max Allowed: {f['max_allowed']}")
        print(f"       Action : {f['recommendation']}")

    print()
    return findings


# =============================================================================
# SECTION 2: Log File Anomaly Detection
# =============================================================================

# --- Anomaly detection rules ---
ANOMALY_RULES = [
    {
        "id": "ANOMALY-001",
        "name": "Repeated Authentication Failures",
        "pattern": r"(?i)(authentication fail|login fail|invalid password|"
                   r"access denied|unauthorized)",
        "threshold": 5,
        "window_minutes": 10,
        "severity": "HIGH",
    },
    {
        "id": "ANOMALY-002",
        "name": "Privilege Escalation Attempts",
        "pattern": r"(?i)(sudo|su\s+root|privilege.escalat|"
                   r"permission.denied.*root|setuid)",
        "threshold": 3,
        "window_minutes": 5,
        "severity": "CRITICAL",
    },
    {
        "id": "ANOMALY-003",
        "name": "Unusual Outbound Connections",
        "pattern": r"(?i)(outbound.*connection|external.*transfer|"
                   r"exfiltrat|curl\s+http|wget\s+http)",
        "threshold": 10,
        "window_minutes": 15,
        "severity": "MEDIUM",
    },
    {
        "id": "ANOMALY-004",
        "name": "Service Crash / Restart Loops",
        "pattern": r"(?i)(segfault|core dump|service.*restart|"
                   r"oom.killer|out.of.memory|fatal error)",
        "threshold": 3,
        "window_minutes": 5,
        "severity": "HIGH",
    },
]


def generate_sample_log_file(path):
    """Generates a realistic sample log file for demonstration.

    Creates entries that include normal operations, authentication failures,
    privilege escalation attempts, and other anomalous patterns.
    """
    base_time = datetime(2026, 6, 30, 8, 0, 0)
    entries = []

    normal_msgs = [
        "INFO  Connection established from 10.0.1.{ip}",
        "INFO  User {user} logged in successfully",
        "INFO  GET /api/v1/data returned 200 in {ms}ms",
        "INFO  Scheduled backup completed for database main_db",
        "DEBUG Health check passed: all services nominal",
        "INFO  POST /api/v1/records returned 201 in {ms}ms",
    ]
    users = ["alice", "bob", "charlie", "diana", "admin"]

    import random
    random.seed(42)

    for i in range(200):
        ts = base_time + timedelta(seconds=i * 15)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        if i in range(30, 38):
            # Cluster of auth failures
            entries.append(
                f"{ts_str} ERROR Authentication failure for user "
                f"'admin' from 192.168.1.{random.randint(100,200)}")
        elif i in (50, 52, 55):
            # Privilege escalation
            entries.append(
                f"{ts_str} WARNING sudo: unauthorized attempt by "
                f"user 'bob' to run command as root")
        elif i in (80, 82, 84, 86):
            # Service restarts
            entries.append(
                f"{ts_str} CRITICAL Service worker-3 segfault at "
                f"0x{random.randint(0,0xFFFF):04x}, core dump generated")
        elif i == 120:
            # Sensitive data in logs (for Section 4)
            entries.append(
                f"{ts_str} DEBUG Processing payment for SSN 123-45-6789, "
                f"card 4532-1234-5678-9012")
        elif i == 150:
            entries.append(
                f"{ts_str} DEBUG API key: AKIAIOSFODNN7EXAMPLE in request headers")
        else:
            msg = random.choice(normal_msgs)
            msg = msg.format(
                ip=random.randint(1, 254),
                user=random.choice(users),
                ms=random.randint(10, 500),
            )
            entries.append(f"{ts_str} {msg}")

    with open(path, 'w') as f:
        f.write("\n".join(entries))

    return path


def analyze_log_anomalies(log_path):
    """Parses a log file and detects anomalies based on predefined rules.

    Args:
        log_path: Path to the log file.

    Returns:
        list of anomaly finding dicts.
    """
    print("=" * 70)
    print("SECTION 2: Log File Anomaly Detection")
    print("=" * 70)
    print(f"\n  Analyzing: {log_path}")

    with open(log_path, 'r') as f:
        lines = f.readlines()

    print(f"  Total log entries: {len(lines)}")
    findings = []

    for rule in ANOMALY_RULES:
        matches = []
        pattern = re.compile(rule["pattern"])
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                matches.append({"line_number": i, "content": line.strip()})

        if len(matches) >= rule["threshold"]:
            findings.append({
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "match_count": len(matches),
                "threshold": rule["threshold"],
                "sample_matches": matches[:3],
            })

    # Print results
    if findings:
        print(f"\n  Anomalies Detected: {len(findings)}\n")
        for f_item in findings:
            print(f"  [{f_item['rule_id']}] {f_item['severity']}: "
                  f"{f_item['rule_name']}")
            print(f"    Occurrences: {f_item['match_count']} "
                  f"(threshold: {f_item['threshold']})")
            print(f"    Sample matches:")
            for m in f_item["sample_matches"]:
                content = m['content'][:80] + "..." if len(m['content']) > 80 else m['content']
                print(f"      Line {m['line_number']}: {content}")
            print()
    else:
        print("\n  No anomalies detected.\n")

    return findings


# =============================================================================
# SECTION 3: S3 Encryption Status Validation (Simulated)
# =============================================================================

def _create_simulated_s3_environment():
    """Creates a simulated AWS S3 environment for demonstration.

    In production, replace this with actual boto3 calls:
        import boto3
        s3 = boto3.client('s3')
        buckets = s3.list_buckets()['Buckets']
    """
    return [
        {
            "Name": "prod-customer-data",
            "CreationDate": "2025-01-15",
            "encryption": {"Rules": [{"ApplyServerSideEncryptionByDefault":
                           {"SSEAlgorithm": "aws:kms"}}]},
            "versioning": "Enabled",
            "public_access_block": True,
            "logging": True,
        },
        {
            "Name": "prod-analytics-raw",
            "CreationDate": "2025-03-22",
            "encryption": {"Rules": [{"ApplyServerSideEncryptionByDefault":
                           {"SSEAlgorithm": "AES256"}}]},
            "versioning": "Enabled",
            "public_access_block": True,
            "logging": True,
        },
        {
            "Name": "dev-temp-uploads",
            "CreationDate": "2025-06-10",
            "encryption": None,  # NOT ENCRYPTED
            "versioning": "Suspended",
            "public_access_block": False,  # PUBLICLY ACCESSIBLE
            "logging": False,
        },
        {
            "Name": "staging-ml-models",
            "CreationDate": "2025-08-05",
            "encryption": {"Rules": [{"ApplyServerSideEncryptionByDefault":
                           {"SSEAlgorithm": "AES256"}}]},
            "versioning": "Enabled",
            "public_access_block": True,
            "logging": False,  # NO LOGGING
        },
        {
            "Name": "backup-db-snapshots",
            "CreationDate": "2024-11-30",
            "encryption": {"Rules": [{"ApplyServerSideEncryptionByDefault":
                           {"SSEAlgorithm": "AES256"}}]},
            "versioning": "Suspended",  # VERSIONING OFF
            "public_access_block": True,
            "logging": True,
        },
    ]


def validate_s3_encryption():
    """Validates encryption and security posture of S3 buckets.

    Uses simulated data. In production, use boto3:
        s3 = boto3.client('s3')
        enc = s3.get_bucket_encryption(Bucket=name)
    """
    print("=" * 70)
    print("SECTION 3: S3 Encryption & Security Validation")
    print("=" * 70)

    buckets = _create_simulated_s3_environment()
    findings = []
    compliant_count = 0

    print(f"\n  Buckets discovered: {len(buckets)}\n")
    header = f"  {'Bucket':<25} {'Encryption':<12} {'Version':<10} {'Public':<8} {'Logging':<8} {'Status'}"
    print(header)
    print("  " + "-" * 85)

    for bucket in buckets:
        name = bucket["Name"]
        issues = []

        # Check encryption
        enc = bucket.get("encryption")
        if enc is None:
            enc_status = "NONE"
            issues.append({
                "check": "encryption",
                "severity": "CRITICAL",
                "detail": "No server-side encryption configured",
                "remediation": (
                    f"aws s3api put-bucket-encryption --bucket {name} "
                    f"--server-side-encryption-configuration "
                    f"'{{\"Rules\":[{{\"ApplyServerSideEncryptionByDefault\":"
                    f"{{\"SSEAlgorithm\":\"AES256\"}}}}]}}'"
                ),
            })
        else:
            algo = enc["Rules"][0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]
            enc_status = algo

        # Check public access
        if not bucket.get("public_access_block"):
            issues.append({
                "check": "public_access",
                "severity": "CRITICAL",
                "detail": "Public access block not enabled",
                "remediation": (
                    f"aws s3api put-public-access-block --bucket {name} "
                    f"--public-access-block-configuration "
                    f"BlockPublicAcls=true,IgnorePublicAcls=true,"
                    f"BlockPublicPolicy=true,RestrictPublicBuckets=true"
                ),
            })

        # Check versioning
        if bucket.get("versioning") != "Enabled":
            issues.append({
                "check": "versioning",
                "severity": "WARNING",
                "detail": "Bucket versioning not enabled",
                "remediation": (
                    f"aws s3api put-bucket-versioning --bucket {name} "
                    f"--versioning-configuration Status=Enabled"
                ),
            })

        # Check logging
        if not bucket.get("logging"):
            issues.append({
                "check": "logging",
                "severity": "MEDIUM",
                "detail": "Access logging not enabled",
                "remediation": f"Enable server access logging for {name}",
            })

        status = "COMPLIANT" if not issues else "NON-COMPLIANT"
        if not issues:
            compliant_count += 1

        ver = bucket.get("versioning", "N/A")
        pub = "Blocked" if bucket.get("public_access_block") else "OPEN"
        log = "Yes" if bucket.get("logging") else "No"

        print(f"  {name:<25} {enc_status:<12} {ver:<10} {pub:<8} {log:<8} {status}")

        if issues:
            findings.append({"bucket": name, "issues": issues})

    # Detailed findings
    print(f"\n  Compliance: {compliant_count}/{len(buckets)} buckets fully compliant\n")

    if findings:
        print("  Detailed Findings:")
        for f_item in findings:
            print(f"\n    Bucket: {f_item['bucket']}")
            for issue in f_item["issues"]:
                print(f"      [{issue['severity']}] {issue['detail']}")
                print(f"        Fix: {issue['remediation']}")

    print()
    return findings


# --- Production boto3 reference (commented) ---
BOTO3_REFERENCE = '''
# === PRODUCTION BOTO3 CODE FOR S3 ENCRYPTION AUDIT ===
import boto3
from botocore.exceptions import ClientError

def audit_s3_encryption_production():
    """Production S3 encryption audit using boto3."""
    s3 = boto3.client('s3')
    buckets = s3.list_buckets()['Buckets']
    findings = []

    for bucket in buckets:
        name = bucket['Name']
        try:
            enc = s3.get_bucket_encryption(Bucket=name)
            rules = enc['ServerSideEncryptionConfiguration']['Rules']
            algo = rules[0]['ApplyServerSideEncryptionByDefault']['SSEAlgorithm']
            if algo != 'AES256' and algo != 'aws:kms':
                findings.append({
                    "bucket": name,
                    "issue": f"Unexpected encryption: {algo}",
                    "severity": "WARNING",
                })
        except ClientError as e:
            if e.response['Error']['Code'] == 'ServerSideEncryptionConfigurationNotFoundError':
                findings.append({
                    "bucket": name,
                    "issue": "No encryption configured",
                    "severity": "CRITICAL",
                })

    return findings
'''


# =============================================================================
# SECTION 4: Sensitive Data Scanner
# =============================================================================

SENSITIVE_DATA_PATTERNS = [
    {
        "id": "PII-001",
        "name": "Social Security Number (SSN)",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "severity": "CRITICAL",
        "regulation": "HIPAA / PCI-DSS",
    },
    {
        "id": "PII-002",
        "name": "Credit Card Number (Visa/MC)",
        "pattern": r"\b(?:4[0-9]{3}|5[1-5][0-9]{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "severity": "CRITICAL",
        "regulation": "PCI-DSS",
    },
    {
        "id": "PII-003",
        "name": "Email Address",
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "severity": "MEDIUM",
        "regulation": "GDPR",
    },
    {
        "id": "SEC-001",
        "name": "AWS Access Key ID",
        "pattern": r"\bAKIA[0-9A-Z]{16}\b",
        "severity": "CRITICAL",
        "regulation": "Internal Security Policy",
    },
    {
        "id": "SEC-002",
        "name": "Generic API Key / Token",
        "pattern": r"(?i)(?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{20,}",
        "severity": "HIGH",
        "regulation": "Internal Security Policy",
    },
]


def scan_sensitive_data(file_path):
    """Scans a file for sensitive data patterns (SSNs, credit cards, keys).

    Args:
        file_path: Path to the file to scan.

    Returns:
        list of sensitive data finding dicts.
    """
    print("=" * 70)
    print("SECTION 4: Sensitive Data Scanner")
    print("=" * 70)
    print(f"\n  Scanning: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    findings = []
    for rule in SENSITIVE_DATA_PATTERNS:
        pattern = re.compile(rule["pattern"])
        for i, line in enumerate(lines, 1):
            for match in pattern.finditer(line):
                matched_text = match.group()
                # Mask sensitive data for the report
                if len(matched_text) > 6:
                    masked = matched_text[:3] + "*" * (len(matched_text) - 6) + matched_text[-3:]
                else:
                    masked = "***"

                findings.append({
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "regulation": rule["regulation"],
                    "line_number": i,
                    "masked_value": masked,
                    "context": line.strip()[:60] + "...",
                })

    # Print results
    if findings:
        by_severity = Counter(f["severity"] for f in findings)
        print(f"\n  Sensitive Data Found: {len(findings)} instances")
        for sev, count in sorted(by_severity.items()):
            print(f"    {sev}: {count}")
        print()

        for f_item in findings:
            print(f"  [{f_item['rule_id']}] {f_item['severity']}: "
                  f"{f_item['rule_name']}")
            print(f"    Line {f_item['line_number']}: {f_item['masked_value']}")
            print(f"    Regulation: {f_item['regulation']}")
            print(f"    Context: {f_item['context']}")
            print()
    else:
        print("\n  No sensitive data patterns detected.\n")

    return findings


# =============================================================================
# SECTION 5: Consolidated Audit Report
# =============================================================================

def generate_audit_report(perm_findings, log_findings, s3_findings, pii_findings):
    """Generates a consolidated JSON and text audit report."""
    print("=" * 70)
    print("SECTION 5: Consolidated Audit Report")
    print("=" * 70)

    report = {
        "report_metadata": {
            "title": "Custom Security Audit Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "auditor": "custom_audit_scripts.py",
            "version": "1.0",
        },
        "summary": {
            "file_permission_violations": len(perm_findings),
            "log_anomalies_detected": len(log_findings),
            "s3_non_compliant_buckets": len(s3_findings),
            "sensitive_data_exposures": len(pii_findings),
            "overall_risk": "HIGH" if any(
                f.get("severity") == "CRITICAL"
                for group in [perm_findings, log_findings, pii_findings]
                for f in group
            ) else "MEDIUM",
        },
        "sections": {
            "file_permissions": perm_findings,
            "log_anomalies": log_findings,
            "s3_encryption": s3_findings,
            "sensitive_data": pii_findings,
        },
    }

    # Text report
    lines = [
        "=" * 60,
        "CUSTOM SECURITY AUDIT REPORT",
        f"Generated: {report['report_metadata']['generated_at']}",
        "=" * 60,
        "",
        "EXECUTIVE SUMMARY",
        f"  Overall Risk Level     : {report['summary']['overall_risk']}",
        f"  Permission Violations  : {report['summary']['file_permission_violations']}",
        f"  Log Anomalies          : {report['summary']['log_anomalies_detected']}",
        f"  S3 Non-Compliant       : {report['summary']['s3_non_compliant_buckets']}",
        f"  Sensitive Data Leaks   : {report['summary']['sensitive_data_exposures']}",
        "",
        "RECOMMENDED ACTIONS",
        "  1. Remediate all CRITICAL S3 encryption findings immediately",
        "  2. Rotate any exposed API keys and credentials",
        "  3. Fix file permissions on private keys and config files",
        "  4. Investigate authentication failure clusters in logs",
        "  5. Implement log redaction for sensitive data fields",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    print(f"\n{report_text}")

    # Save JSON report
    json_path = os.path.join(OUTPUT_DIR, 'custom_audit_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  JSON report saved to: {json_path}")

    # Save text report
    txt_path = os.path.join(OUTPUT_DIR, 'custom_audit_report.txt')
    with open(txt_path, 'w') as f:
        f.write(report_text)
    print(f"  Text report saved to: {txt_path}\n")

    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  Custom Audits with Python - Security Compliance Verification    #")
    print("#" * 70 + "\n")

    # 1. File Permission Audit (scan current project directory)
    scan_target = os.path.dirname(os.path.abspath(__file__))
    perm_findings = audit_file_permissions(scan_target)

    # 2. Generate sample log and analyze for anomalies
    sample_log = os.path.join(OUTPUT_DIR, 'sample_application.log')
    generate_sample_log_file(sample_log)
    log_findings = analyze_log_anomalies(sample_log)

    # 3. S3 Encryption Validation (simulated)
    s3_findings = validate_s3_encryption()

    # 4. Sensitive Data Scanner (scan the sample log)
    pii_findings = scan_sensitive_data(sample_log)

    # 5. Consolidated Report
    generate_audit_report(perm_findings, log_findings, s3_findings, pii_findings)

    print("\n" + "#" * 70)
    print("#  Audit Complete - All reports saved to output directory.         #")
    print("#" * 70 + "\n")

"""
NexGaurd - MITRE ATT&CK Threat Mapping and Proactive Defence Playbooks.
Maps predicted network world states and attack classifications to standard MITRE ATT&CK tactics,
techniques, and actionable containment recommendations.
"""

MITRE_TECHNIQUES = {
    0: {
        "tactic": "Baseline Operation",
        "technique_id": "T0000",
        "technique_name": "Normal Enterprise Traffic",
        "description": "Network telemetry exhibits nominal transition dynamics with balanced TCP/UDP ratios and stable port entropy.",
        "kill_chain_phase": "Normal Operation",
        "risk_level": "LOW",
        "action_playbook": [
            "Maintain baseline behavioral telemetry logging.",
            "Periodically verify TLS certificate configurations and egress boundaries.",
            "No containment action required."
        ]
    },
    1: {
        "tactic": "Credential Access / Initial Access",
        "technique_id": "T1110.001",
        "technique_name": "Brute Force: Password Guessing (FTP Port 21)",
        "description": "Attacker is iteratively attempting authentication combinations against the FTP service at high velocity before access is confirmed.",
        "kill_chain_phase": "Initial Access",
        "risk_level": "HIGH",
        "action_playbook": [
            "Proactively rate-limit inbound authentication requests on port 21.",
            "Enforce IP reputation blacklisting on originating IP subnets.",
            "Trigger automated credential rotation for active administrative accounts.",
            "Prepare automated fail2ban firewall rules before connection threshold breaches."
        ]
    },
    2: {
        "tactic": "Credential Access / Initial Access",
        "technique_id": "T1110.001 / T1021.004",
        "technique_name": "Brute Force: SSH Remote Services (Port 22)",
        "description": "Attacker is initiating rapid credential spray against SSH daemon. Pre-compromise progression signals an imminent valid session spawn.",
        "kill_chain_phase": "Initial Access & Privilege Escalation",
        "risk_level": "CRITICAL",
        "action_playbook": [
            "Temporarily isolate or challenge SSH port 22 access via Web/Zero-Trust proxy.",
            "Enforce mandatory hardware-token MFA challenge on subsequent SSH handshakes.",
            "Monitor auth.log / systemd journal for rapid pam_unix failures.",
            "Deploy preemptive tarpit / TCP throttling on aggressive source addresses."
        ]
    }
}

RECON_INDICATOR = {
    "tactic": "Reconnaissance",
    "technique_id": "T1046",
    "technique_name": "Network Service Discovery / Port Scan",
    "description": "Elevated port entropy and abnormal SYN/ACK distribution detected in temporal sequence, indicating systematic network reconnaissance.",
    "kill_chain_phase": "Reconnaissance",
    "risk_level": "MEDIUM",
    "action_playbook": [
        "Enable adaptive IDS signature throttling for scanning source IP.",
        "Block ICMP/SYN port probes at perimeter gateway.",
        "Verify internal honeypot telemetry for decoy interactions."
    ]
}

def get_mitre_intel(class_idx, port_entropy=0.0, syn_ratio=0.0):
    """
    Returns detailed MITRE ATT&CK intelligence based on predicted class
    and anomalous state markers.
    """
    base_info = MITRE_TECHNIQUES.get(class_idx, MITRE_TECHNIQUES[0]).copy()
    if class_idx == 0 and (port_entropy > 2.0 or syn_ratio > 0.4):
        recon = RECON_INDICATOR.copy()
        recon["notes"] = f"Anomalous port entropy ({port_entropy:.2f}) and SYN ratio ({syn_ratio*100:.1f}%) observed during baseline activity."
        return recon
    return base_info

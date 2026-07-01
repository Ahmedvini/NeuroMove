# Security Policy

**NeuroMove** — *Design & Development of a Brain–Computer Interface for Assistive Motion Control,
Rehabilitation & Mobility Applications* (E-JUST graduation project).

## ⚠️ Research status — read first

NeuroMove is an **academic research and educational project**. It is **not** a certified medical
device and **not** a production security product. Nothing here has undergone independent security
certification or clinical validation.

This repository includes security-flavoured components — a hardware cryptographic stack
(`fpga/rtl/security/`: AES-256-GCM, SHA-256, HMAC-SHA256, RSA-2048, a secure-boot skeleton) and an
EEG biometric identification/authentication system (`Secure-EEG-…-main/`). These are **research
demonstrations of the techniques**, not audited, hardened, or side-channel-resistant
implementations.

**Do not** rely on any component here to protect real patient data, real biometric identity, or a
real assistive/prosthetic device without an independent security review and appropriate regulatory
clearance.

## Supported versions

This is an actively developed graduation project. Only the current `main` branch is maintained;
there are no long-term supported releases.

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| tagged snapshots / forks | ❌ |

## Reporting a vulnerability

If you discover a security issue — in the crypto cores, the biometric auth pipeline, the FastAPI
service (`simulation/coppeliasim/`), or anywhere else — please report it **privately** rather than
opening a public issue:

- **Email:** ahmed.elsheikh@ejust.edu.eg
- Alternatively, use GitHub's **private vulnerability reporting** ("Report a vulnerability" under the
  repository's *Security* tab), if enabled.

Please include:

1. The affected component and file(s)/module(s).
2. A description of the issue and its potential impact.
3. Steps to reproduce (proof-of-concept, inputs, or a failing test if possible).
4. Any suggested remediation.

**What to expect:** because this is a student research project, responses are best-effort — we aim to
acknowledge a report within **~7 days**. Please give us reasonable time to investigate and address
the issue before any public disclosure (coordinated disclosure).

## Scope

**In scope:** cryptographic correctness bugs in `fpga/rtl/security/`; authentication/identification
bypasses in `Secure-EEG-…-main/`; injection, auth, or data-exposure issues in the
`simulation/coppeliasim/` FastAPI service and its Docker deployment; secret/credential leakage in the
repo history.

**Out of scope:** the deliberate research limitations noted above (e.g. lack of side-channel
hardening, absence of medical/production certification); vulnerabilities in third-party datasets,
upstream models (e.g. DB-ATCNet), or tooling (Vivado, COMSOL, CoppeliaSim); missing hardening on
example/demo configuration and the `TODO` dataset-path integration hooks in `fpga/sim/` and
`fpga/rtl/security/eeg_dataset_config.sv`.

## Handling sensitive data

- **Never commit** real EEG/fNIRS recordings, personal biometric data, private keys, or credentials.
  Raw datasets are git-ignored and downloaded per [`data/README.md`](data/README.md).
- Treat any EEG data as sensitive personal data; use only properly licensed, de-identified public
  datasets in this repository.

---

*Maintained by the NeuroMove team at Egypt-Japan University of Science and Technology (E-JUST),
supervised by Dr. Reda Albassiouny & Dr. Sameh Sherif. Licensed under GNU GPL v3 — see
[LICENSE](LICENSE).*

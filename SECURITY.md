# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in widemem, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email **radu@cioplea.com** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

You should receive a response within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

widemem is a local-first library. Security concerns include:

- **Data storage** — Memories are stored in local SQLite and FAISS files. Ensure proper file permissions on `~/.widemem/` and any custom storage paths.
- **LLM API keys** — Keys are passed via config or environment variables. widemem does not log, transmit, or store API keys beyond what the underlying provider SDKs do.
- **Prompt injection** — User-provided text is sent to LLMs for extraction and conflict resolution. If you're processing untrusted input, be aware that adversarial text could influence extraction results.
- **YMYL is not a security boundary** — The YMYL classifier uses keyword matching for prioritization. It is not a content filter, access control mechanism, or compliance tool.

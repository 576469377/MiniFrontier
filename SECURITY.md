# Security

A concrete private reporting address has not yet been supplied by the maintainers. Public release is pending this contact detail; see [release preparation](docs/releases/v0.1.0.md). Do not include secrets or exploit payloads in public issues.

- Do not execute untrusted Python model code, plugins or generated commands.
- Do not load unknown pickle checkpoints. Prefer safetensors for eventual published weights.
- Keep datasets, credentials, tokens, private paths and large training artifacts out of Git and release bundles.
- The retained upstream snapshots are reference material. Importing a snapshot may require dependencies and capabilities not provided by MiniFrontier.
- Dependency pinning and source-oracle tests do not make third-party code or model outputs trustworthy.
- GPU selection is a point-in-time check, not an exclusive resource reservation; memory use must still be monitored.

The browser demo binds to 127.0.0.1 by default and lists only locally discovered checkpoints with passing, weight-bound capability evidence by default. It is intended for local use or SSH forwarding; it has no public-service authentication. Checkpoints are loaded with torch.load(weights_only=True), and generated text is rendered as text content. The demo does not execute generated code.

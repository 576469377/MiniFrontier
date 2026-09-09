# Security

No private reporting contact is currently listed in this repository. General bugs without sensitive details can be reported in [Issues](https://github.com/576469377/MiniFrontier/issues). Please keep credentials, personal data and exploit details out of public reports. This page will list the private channel when it is available.

MiniFrontier currently provides research code and local training tools. Use checkpoints from known sources and review code before executing it. The loaders use `torch.load(weights_only=True)`; the project has not published model weights or completed a security audit of a deployed service.

The browser demo binds to `127.0.0.1` by default and is intended for local use or SSH forwarding. It has no public-service authentication. Generated text is displayed as text, and the demo does not execute generated code. The separate training verifier can run code tasks in its restricted worker; see the [posttraining guide](docs/guides/posttraining-adaptation.md) for that scope.

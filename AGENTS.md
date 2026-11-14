# Agent instructions

**Engineering architecture and reliability:** Adopt [GCL-ENGINEERING-001 v1.0.0](https://github.com/grandchallenge/gcl-standards/blob/2a3cfed9419ba9b74eeaaee70b9db8c3f47f3c9b/standards/GCL-ENGINEERING-001.md), pinned at protected `gcl-standards` commit `2a3cfed9419ba9b74eeaaee70b9db8c3f47f3c9b`. Engineering agents own architecture, necessity, integration, tests, deploy/readback and retirement; reuse GitHub and existing components before inventing infrastructure. Do not duplicate workflow authority or mistake CI/merge for live completion. Preserve this repository's superior, reserved, and domain-specific policies. Cross-repository adoption requires explicit readback, never assumption.

MODULUS is a provider repository and has no mathematical, certification, or
claim-promotion authority. Under `GCL-AGENT-STAFFING-001` version `1.0.0`, one
Codex system may implement, run bounded experiments, staff distinct
non-reserved logical audit passes, merge through satisfied protected controls,
and read back the result. Authoring-system Adversary and Referee passes are
`non_authoring_read_only`. Automation may not write directly to protected
branches or promote benchmark results beyond admitted evidence.

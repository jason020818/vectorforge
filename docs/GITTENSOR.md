# Gittensor registration

VectorForge is intended as a public SN74 contribution arena: miners open PRs
against documented issues, and maintainers merge only work that keeps
correctness and reproducibility intact.

This file is the maintainer checklist. It does not install the GitHub App and
does not submit the registration form.

## Why this repository fits

- Public MIT repository: `https://github.com/jason020818/vectorforge`
- Active C++/Python engine with CI, tests, and a documented contribution path
- Real ANN value: exact `FlatIndex` oracle plus paper-faithful `HNSWIndex`
- Phase 3A fair benchmark infrastructure is in tree; official Phase 3B numbers
  are not claimed until a controlled native Linux run exists
- Issues must be measurable. Performance issues require a recorded baseline

## Maintainer registration steps

These two steps cannot be completed from the git tree. They require GitHub
admin access on `jason020818/vectorforge`.

1. Install the Gittensor Mirror GitHub App (read-only) on this repository:
   https://docs.gittensor.io/register-repository.html
2. Submit the form at https://gittensor.io/repository-registration

Suggested form values:

- Repository URL: `https://github.com/jason020818/vectorforge`
- Short description: `CPU ANN retrieval engine with a public, reproducible performance competition framework`
- GitHub handle: `jason020818`

## After approval

- Miners should pick an open issue, keep PRs small, and follow CONTRIBUTING.md
- Maintainers continue to merge or close PRs. Gittensor never merges for us
- Self-merged PRs are not scored unless there is external review approval
- Do not open PERF issues until an official Phase 3B baseline is recorded

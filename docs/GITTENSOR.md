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
- Do not open PERF issues until an official Phase 3B baseline is recorded

## Scoring (current Gittensor rules)

These are the official SN74 distinctions. They are not VectorForge-specific
policy, and they do not imply a live Gittensor config for this repository.

1. PRs authored by repository maintainers whose GitHub association is
   `OWNER`, `MEMBER`, or `COLLABORATOR` are ignored for PR-side OSS scoring.
2. For ordinary non-maintainer miner PRs, a self-merge is rejected unless
   the PR has at least one external approval.
3. Maintainer earnings on their own repository, when configured by
   Gittensor, come through the repository's `maintainer_cut` rather than
   through PR-side or issue-discovery rewards.
4. Do not invent a `maintainer_cut` value for VectorForge. VectorForge is
   not yet registered, so no live repository config exists.
5. Phase 3B is deferred. There is no official competitor performance claim.

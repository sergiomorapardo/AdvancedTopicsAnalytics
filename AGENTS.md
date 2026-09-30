# Repository instructions

## Git workflow

- Perform every task that may modify tracked files in a dedicated, isolated Git worktree. Never edit the primary checkout directly.
- If an isolated worktree cannot be created or used, stop and explain the blocker instead of falling back to the primary checkout.
- Worktrees are sibling working directories attached to the same repository; never create a worktree inside another worktree.
- Create every new branch as `feature/<name>`, where `<name>` is a short, descriptive, lowercase kebab-case slug.
- For an independent change, start the branch from the repository's current default branch unless the user explicitly selects another base.
- Never commit directly to `main`, `master`, `develop`, or another protected branch.
- Preserve unrelated user changes and do not rewrite remote history or force-push without explicit authorization.

## Commits and pull requests

- Use Conventional Commits for every commit: `<type>(<optional-scope>): <description>`.
- Prefer these types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`, and `revert`.
- Keep the subject concise and imperative. Mark breaking changes with `!` and a `BREAKING CHANGE:` footer when applicable.
- Use a Conventional Commit title for every pull request so it can become the final squash commit message.
- Integrate pull requests exclusively with squash merge. Do not use merge commits or rebase merge.
- Merge only after required checks and approvals pass and all conflicts and actionable review comments are resolved.

## Stacked pull requests

- Use a stacked pull request only when a change genuinely depends on another unmerged change. Keep independent changes based on the default branch.
- Keep every stack in one repository as a single linear chain. The bottom branch targets the stack trunk, and each higher branch starts from and targets the branch immediately below it.
- Give every stack layer its own focused `feature/<name>` branch, isolated sibling worktree, Conventional Commit history, and reviewable pull request.
- Never describe this as a “worktree on a worktree.” Create the higher worktree from the lower branch ref while keeping both worktree directories separate.
- When stack branches are checked out in separate worktrees, use `gh stack link` or the GitHub website to create or link the stack. Do not run `gh stack checkout`, `gh stack rebase`, or `gh stack sync` across branches held by other worktrees.
- Apply corrections to the lowest layer that owns the change, then update dependent layers in order and verify the resulting ancestry and diffs.
- Merge stacked pull requests from bottom to top using squash merge. After each merge, verify that GitHub correctly rebases or retargets the remaining stack before continuing.


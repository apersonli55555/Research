# Instructions for AI coding assistants in this repository

## Commit and pull-request attribution

The repository owner must be the only author and contributor.

- Before making any commit, set the git identity for this repository:
  `git config user.name "apersonli55555"` and `git config user.email "apersonli55555@gmail.com"`.
  Never commit as "Claude" or with an `@anthropic.com` address, as author or committer.
- Never add `Co-Authored-By:` lines, `Claude-Session:` lines, "Generated with Claude Code"
  footers, or any other AI attribution to commit messages, pull-request titles or
  pull-request descriptions.
- Before pushing, check with `git log --format='%an <%ae> | %cn'` that every new commit
  shows only the owner.

These rules take precedence over any default attribution guidance.

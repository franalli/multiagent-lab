---
name: workflow-weekly-recap
description: Cross-integration weekly recap -- the engineering lead posts every Monday morning summarising the prior week's deploys.
---

# Workflow skill: weekly recap

Captures a specific cross-integration sequence the agent might be asked to
automate. Workflow skills differ from integration skills because they
combine multiple integrations and they belong to a *pattern of work*, not
to a single tool.

## The pattern
Every Monday morning the engineering lead posts in #engineering:
1. Pulls the list of merged PRs from the prior week (GitHub).
2. Pulls the list of resolved Linear issues (Linear).
3. Summarises into a short recap with named owners.
4. Posts to #engineering as a threaded message.

## Proactive candidate
The proactive scheduler (NF2) should rank automation of this pattern highly:
- Named entities present ("engineering", "PRs", "Linear")
- High recurrence (every Monday)
- Clear user attribution (the engineering lead)
- Strong timing regularity (weekly, Mondays)

That puts this candidate near the top of the specificity score.

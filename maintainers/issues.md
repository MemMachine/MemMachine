
# Issue Management for Maintainers

As a project maintainer, effectively managing issues is crucial for maintaining a healthy and active repository. This document outlines the key tasks involved.

## Triage and Labeling

When a new issue is created, the first step is to **triage** it. This involves reviewing the issue and assigning appropriate labels to categorize it. This helps other contributors understand the issue at a glance and makes it easier to find related topics.

The issue templates set each issue's type: `Bug`, `Feature`, `Documentation`, or `Question`. Correct the type when an issue was filed under the wrong one.

**Common Labels to Use:**

- `documentation`: The issue relates to documentation updates.
- `good first issue`: An issue that is well-defined and a good starting point for new contributors.

You can learn more about managing your labels in the [GitHub documentation](https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels "null").

### Reproducing and Validating

Before a bug can be fixed, a maintainer should attempt to **reproduce** it. This involves following the steps provided by the user to confirm the bug exists and to understand its scope. If you can't reproduce the bug, you may need to ask the user for more information, such as the environment they're using or the exact steps they took.

### Closing Issues

Keeping the issue tracker clean is a key part of maintenance. You can close issues for several reasons:

- **Fixed:** The issue has been resolved by a merged pull request.
- **Duplicate:** The issue is a duplicate of an existing one. In this case, close the new issue and link to the original.
- **Stale:** The issue has been open for a long time with no activity. The stale workflow (`.github/workflows/stale.yml`) labels idle issues `Stale` and later closes them; add the `keep-open` label to exempt one.
- **Won't Fix:** You have decided not to implement the requested feature or fix the bug. In this case, it's important to provide a brief explanation.

### Prioritizing the Backlog

With a growing list of issues, it's important to **prioritize** them. You can use labels like `priority: high`, `priority: medium`, and `priority: low` to help organize your backlog. You can also use [GitHub Milestones](https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/about-milestones "null") to group issues that should be completed by a specific date or for a particular release.

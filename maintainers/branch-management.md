
# Branch Management

This document outlines the branching strategy used for this project, including how to create new branches, handle merge conflicts, and keep your local environment up to date.

## Branching Strategy: Trunk-Based Development

For this project, we use a **Trunk-Based Development** branching strategy. This model is characterized by a single, shared `main` branch that is always ready to be released. This approach is widely regarded as the best practice for large and growing open-source projects, as it promotes rapid integration, reduces complexity, and minimizes merge conflicts.

Here's how it works:

1. **The `main` branch is the "trunk".** Changes merge into `main`, directly or, for a large effort, through a shared feature branch that merges into `main` later. `main` should always be stable and in a deployable state.

2. **Use short-lived feature branches.** Instead of working directly on `main`, developers create short-lived branches for new features or bug fixes. These branches should be merged into `main` as soon as the work is complete, ideally within a few hours or a day. This ensures the changes are small, easy to review, and less likely to cause complex merge conflicts.

3. **Use Pull Requests (PRs) for all merges.** No code should be committed directly to `main`. Every change, no matter how small, must be submitted as a pull request for review and approval by at least two maintainers.

## How to Keep Your Local Branch Up-to-Date

To prevent your local feature branch from "diverging" too far from the `main` branch, you should regularly update it with the latest changes. This minimizes the risk of merge conflicts when you are ready to merge your PR.

1. **Switch to your `main` branch.**

    ```bash
    git checkout main
    ```

2. **Pull the latest changes.** This fetches the new commits from the upstream repository's `main` branch.

    ```bash
    git pull upstream main
    ```

3. **Switch back to your feature branch.**

    ```bash
    git checkout <your-feature-branch>
    ```

4. **Rebase your feature branch.** This is the key step. Rebasing rewrites your commit history to place your commits on top of the latest `main` branch commits, resulting in a clean, linear history.

    ```bash
    git rebase main
    ```

### Handling Merge Conflicts

Merge conflicts are inevitable in a collaborative environment. They happen when two different changes are made to the same part of a file.

1. **Run `git status`**. This will show you which files have conflicts.

    ```bash
    git status
    ```

2. **Open the conflicted files.** Your code editor will highlight the conflicted areas. You will see sections of code marked with `<<<<<<<`, `=======`, and `>>>>>>>`.

3. **Resolve the conflict.** Manually edit the file to decide which changes to keep. Delete the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).

4. **Add the resolved file.** Once you have resolved all conflicts in a file, stage it to notify Git that it's resolved.

    ```bash
    git add <path/to/resolved/file>
    ```

5. **Continue the rebase.** Once all conflicts are resolved and staged, you can continue the rebase.

    ```bash
    git rebase --continue
    ```

6. **Push your changes.** You will likely need to force-push your branch after a rebase.

    ```bash
    git push --force-with-lease
    ```

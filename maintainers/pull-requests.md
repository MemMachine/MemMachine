# Reviewing Pull Requests

As a maintainer, you play a vital role in ensuring the quality and integrity of the codebase by reviewing pull requests (PRs). A good review process not only checks for bugs but also helps to maintain code quality and provides constructive feedback to contributors.

## The Pull Request Workflow

1. **Check out the PR locally.** While you can review the code directly on GitHub, the best way to validate changes is to run and test them yourself. This is especially important for PRs from external contributors, which come from a forked repository.
2. **Provide constructive feedback.** Leave comments on specific lines of code or files to suggest improvements. Remember to be friendly and encouraging.
3. **Ensure code quality.** Check that the code adheres to the project's style guide and best practices.
4. **Merge the PR.** Once the code has been tested, all checks have passed, and you are satisfied with the changes, you can approve and merge the PR.

## Important Note for Reviewers

As a reviewer, you must have your own fork of the upstream repository. Clone your fork to your local machine to ensure you have the correct permissions for pushing changes, creating branches, and testing PRs. This setup is required for contributing and reviewing effectively.

## Setting up a Contributor's Fork Locally

The process of testing a contributor's code from a fork requires a couple of extra steps. This method is the best way to get their changes onto your machine without needing to clone their entire repository.

For these steps, you will need the pull request number and the contributor's GitHub username. You can find both of these on the pull request page.

1. **Add the contributor's repository as a new remote.** A "remote" is a URL that points to a Git repository. You already have a remote for your main repository (usually called `origin`). Now, you will add one for the contributor's fork.

    You can use one of the following methods:

    Note: Replace `<contributor-username>` with the contributor's GitHub username and `<repo-name>` with the name of your repository.

    - **HTTPS method:**

      ```bash
      git remote add <contributor-username> https://github.com/<contributor-username>/<repo-name>.git
      ```

    - **SSH method:**  
      You must have SSH keys synced with GitHub for this to work.

      ```bash
      git remote add <contributor-username> git@github.com:<contributor-username>/<repo-name>.git
      ```

2. **Fetch the branch from the new remote.** This command will download all the branches from the contributor's fork to your local machine.

    ```bash
    git fetch <contributor-username>
    ```

3. **Check out the contributor's branch.** You can now create a new local branch that is a copy of their branch.

    ```bash
    git checkout -b <local-branch-name> <contributor-username>/<contributor-branch-name>
    ```

    - `local-branch-name`: a descriptive name for your new local branch (e.g., `fix-issue-123`).
    - `contributor-branch-name`: the name of the branch the contributor is working on (this is often the same as `fix-issue-123` or similar).

4. **Test the changes.** You now have a local branch with the contributor's changes and can test the code as needed.

5. **Clean up.** When you are done, you can switch back to your `main` branch and delete the temporary branch you created.

    ```bash
    git checkout main
    git branch -d <local-branch-name>
    ```

    You can also remove the temporary remote you created:

    ```bash
    git remote remove <contributor-username>
    ```

### Providing Feedback and Merging

Once you've tested the changes and are ready to provide feedback, you can do so directly on the pull request page on GitHub. If you are satisfied with the changes, you can approve the pull request.

**Note:** The protected branch rules for this repository require a minimum of 2 approving reviews before the "Merge" button becomes active. You will not be able to merge the pull request until it has met this requirement.

Once all required reviews have been completed, you can merge the pull request.

You can learn more about this process and other advanced topics in the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests "null").

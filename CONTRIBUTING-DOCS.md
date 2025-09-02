# Contributing to MemMachine Documentation

Welcome! Your contributions to the MemMachine documentation are highly valued.
This guide will walk you through the process of setting up your local
environment and making changes.

## 1. Getting Started: The Issue-First Workflow

Before you begin, please check our
[issues page](https://github.com/MemMachine/MemMachine/issues) to see if the
issue has already been reported.

For a new documentation feature or a major change, please create a new issue
first. This allows us to discuss the change and ensures that no one else is
already working on it. When creating an issue, please use the
**Documentation Issue** template.

For small, straightforward fixes like typos or broken links, you are welcome to
create a pull request directly without filing an issue first.

## 2. Local Setup

Follow these steps to set up your local environment:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/MemMachine/MemMachine.git
    cd MemMachine
    ```

2. **Install the Mintlify CLI:**

    Use `sudo` to install the package with elevated privileges:

    ```bash
    sudo npm i -g mintlify
    ```

    Alternatively, you can change npm's default global package directory to a
    location in your home directory where you have write permissions:

    ```bash
    mkdir ~/.npm-local
    npm config set prefix '~/.npm-local'
    ```

    Add this directory to your PATH by adding the following line to your
    `~/.bashrc` or `~/.zshrc` file:

    ```bash
    export PATH=~/.npm-local/bin:$PATH
    ```

    Then, reload your terminal configuration:

    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

    After this, you can install global packages without needing sudo:

    ```bash
    npm install -g mintlify
    ```



3. **Install Dependencies:** Run the following command at the root of your
documentation (where `mint.json` is located).

    ```bash
    mintlify install
    ```

    This command will re-install any necessary dependencies for the project.

4. **Serve the Documentation:**

    ```bash
    mintlify dev
    ```

    This command will start a local web server. You can view your changes live
    by navigating to the URL provided in the terminal (usually
    `http://localhost:3000`).

## 3. Making a Contribution

- Find the file you want to edit in the `docs/` directory.
- Make your changes.
- Save the file.
- Preview your changes in your browser.

## 4. Submitting a Pull Request

Once your changes are complete and you have verified them locally, follow these
steps to submit a pull request:

1. **Create a New Branch:**

    ```bash
    git checkout -b docs/your-feature-name
    ```

2. **Commit Your Changes:**

    ```bash
    git add .
    git commit -sS -m "Docs: Update the getting started guide"
    ```

    **Remember to use the `-sS` flags to sign your commit.** Unsigned commits
    will fail the GitHub Actions checks.

3. **Push the Branch:**

    ```bash
    git push origin docs/your-feature-name
    ```

4. **Open a Pull Request:** Go to the GitHub repository page and open a pull
  request. Complete the Pull Request template and submit the pull request.

Thank you for your contribution!

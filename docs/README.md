# Local Documentation Setup Guide

This repository contains the source files for our Mintlify documentation site. Follow the instructions below to set up a local development environment to write, preview, and test documentation changes before pushing them live.

---

## Prerequisites

Before starting, ensure you have **Node.js (v18 or higher)** and **Git** installed on your system.

### macOS
1. **Node.js:** Install via [nodejs.org](https://nodejs.org/) or using Homebrew:
   ```bash
   brew install node
   ```
2. **Git:** Comes pre-installed on macOS, or install via Homebrew:
   ```bash
   brew install git
   ```

### Windows
1. **Node.js:** Download and run the Windows Installer (`.msi`) from [nodejs.org](https://nodejs.org/). Ensure the option to **"Add to PATH"** is selected during installation.
2. **Git:** Download and install [Git for Windows](https://gitforwindows.org/). We strongly recommend using **Git Bash** or **PowerShell** for terminal commands.

---

## Step-by-Step Setup

### 1. Install the Mintlify CLI

Open your terminal (Terminal on macOS, Git Bash or PowerShell on Windows) and install the official CLI globally:

```bash
npm install -g mint
```

> **Windows Permission Note:** If running PowerShell gives an execution policy error (`ps1 cannot be loaded`), run PowerShell as an Administrator and execute:
> ```powershell
> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

> **macOS Permission Note:** If you encounter `EACCES` permission errors when installing globally, run:
> ```bash
> sudo npm install -g mint
> ```

---

### 2. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY_NAME>
```

---

### 3. Start the Local Preview Server

Navigate to the project root directory (where `docs.json` or `mint.json` is located) and start the local server:

```bash
mint dev
```

* The local preview will automatically open at `http://localhost:3000`.
* **Hot Reloading:** Edits made to any `.mdx` file will immediately update in your browser upon saving.

---

### 4. Authenticate for Full Features (Optional)

To enable local search indexing and AI assistant features during local development, authenticate your CLI:

```bash
mint login
```

---

## Helpful Commands & Tools

| Command | Purpose |
| :--- | :--- |
| `mint dev` | Starts the local preview server on `localhost:3000`. |
| `mint dev --port 3001` | Starts the server on a custom port if `3000` is in use. |
| `mint broken-links` | Scans the local repository and reports any broken internal/external links. |
| `mint update` | Updates the Mintlify CLI to the latest version. |

### Recommended Code Editor Setup
* **VS Code / Cursor:** Install the **Mintlify MDX** extension for snippet auto-completion and component previews.
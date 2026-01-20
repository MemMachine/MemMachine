"""MemMachine installation and configuration script."""

import logging
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import psutil
import requests

from memmachine import setup_nltk
from memmachine.installation.configuration_wizard import (
    ConfigurationWizard,
)
from memmachine.installation.utilities import (
    DEFAULT_NEO4J_PASSWORD,
    LINUX_JDK_TAR_NAME,
    LINUX_JDK_URL,
    LINUX_NEO4J_TAR_NAME,
    LINUX_NEO4J_URL,
    MACOS_JDK_TAR_NAME_ARM64,
    MACOS_JDK_TAR_NAME_X64,
    MACOS_JDK_URL_ARM64,
    MACOS_JDK_URL_X64,
    NEO4J_DIR_NAME,
    NEO4J_WINDOWS_SERVICE_NAME,
    WINDOWS_JDK_URL,
    WINDOWS_JDK_ZIP_NAME,
    WINDOWS_NEO4J_URL,
    WINDOWS_NEO4J_ZIP_NAME,
    ScriptType,
)

logger = logging.getLogger("MemMachineInstaller")


def get_memmachine_config_dir() -> str:
    """Get the MemMachine configuration directory path."""
    return str(Path("~/.config/memmachine").expanduser())


class Installer(ABC):
    """Abstract base class for MemMachine installers."""

    def __init__(self) -> None:
        """Initialize Installer with default installation directory."""
        self.install_dir = str(Path("memmachine-dependencies").resolve())

    @abstractmethod
    def install_and_start_neo4j(self) -> None:
        """Install and start Neo4j Community Edition."""

    @abstractmethod
    def check_neo4j_running(self) -> bool:
        """Check if Neo4j is running."""

    @abstractmethod
    def get_run_script_type(self) -> ScriptType:
        """Get the script type for the installer."""

    def ask_install_dir(self) -> None:
        """Prompt user for Neo4j installation directory."""
        if not Path(self.install_dir).exists():
            logger.info(
                "Installation directory %s does not exist. Creating...",
                self.install_dir,
            )
            Path(self.install_dir).mkdir(parents=True, exist_ok=True)
        else:
            logger.info("Using installation directory: %s", self.install_dir)

    def install(self, prompt: bool = True) -> None:
        """Install and configure MemMachine."""
        neo4j_started_by_installer = False
        if not self.check_neo4j_running():
            choice = (
                input(
                    "Cannot find Neo4j locally. Do you want to install and start Neo4j? (y/n): "
                )
                .strip()
                .lower()
            )
            if choice == "y":
                self.install_and_start_neo4j()
                neo4j_started_by_installer = True
        else:
            logger.info("Neo4j is already running.")

        wizard_args = ConfigurationWizard.Params(
            neo4j_provided=neo4j_started_by_installer,
            destination=get_memmachine_config_dir(),
            prompt=prompt,
        )
        wizard = ConfigurationWizard(wizard_args)
        wizard.run_wizard()

        logger.info("MemMachine installation and configuration completed.")


class LinuxEnvironment:
    """Environment utilities for Linux installation."""

    def download_file(self, url: str, dest: str) -> None:
        """Download file on Linux."""
        urllib.request.urlretrieve(url, dest)

    def extract_tar(self, tar_path: str, extract_to: str) -> None:
        """Extract a .tar.gz file on Linux or macOS."""
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)

    def start_neo4j(self, java_home: str, neo4j_dir: str) -> None:
        logger.info("Starting Neo4j with JAVA_HOME=%s, dir=%s...", java_home, neo4j_dir)

        # Build environment
        env = os.environ.copy()
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{java_home}/bin:" + env["PATH"]
        env["NEO4J_AUTH"] = "neo4j/neo4j"  # Required so we can set initial password

        neo4j_bin = Path(neo4j_dir) / "bin"

        # 1️⃣ Set initial password BEFORE starting the server
        logger.info("Setting initial Neo4j password...")
        subprocess.run(
            [
                str(neo4j_bin / "neo4j-admin"),
                "dbms",
                "set-initial-password",
                DEFAULT_NEO4J_PASSWORD,
            ],
            env=env,
            check=True,
        )
        logger.info("Initial password set successfully.")

        # 2️⃣ Start the Neo4j server
        logger.info("Starting Neo4j server...")
        subprocess.run(
            [str(neo4j_bin / "neo4j"), "start"],
            env=env,
            check=True,
        )
        logger.info("Neo4j server started.")

    @staticmethod
    def check_neo4j_running() -> bool:
        url = "http://127.0.0.1:7474/"
        try:
            response = requests.get(url, timeout=5)
        except requests.RequestException:
            logger.exception("Failed to connect to Neo4j.")
            return False
        else:
            return response.status_code == 200


class LinuxInstaller(Installer):
    """Installer for Linux using system package managers."""

    def __init__(self, environment: LinuxEnvironment | None = None) -> None:
        """Initialize LinuxInstaller with default installation directory."""
        super().__init__()
        self.environment = environment or LinuxEnvironment()

    def get_run_script_type(self) -> ScriptType:
        """Get the script type for Linux."""
        return ScriptType.BASH

    def check_neo4j_running(self) -> bool:
        """Check if Neo4j service is running on Linux."""
        return self.environment.check_neo4j_running()

    @staticmethod
    def find_jdk21_dir(install_dir: str) -> Path:
        """
        Find the JDK 21 directory whose name starts with 'jdk-21'.

        Falls back to install_dir/jdk-21 if nothing found.
        """
        base = Path(install_dir)

        for entry in base.iterdir():
            if entry.is_dir() and entry.name.startswith("jdk-21"):
                return entry

        # fallback if not found
        return base / "jdk-21"

    def install_and_start_neo4j(self) -> None:
        """Install and start Neo4j Community Edition on Linux."""
        logger.info("Installing Neo4j Community Edition...")
        self.ask_install_dir()

        # Install OpenJDK 21
        logger.info("Downloading and installing %s...", LINUX_JDK_TAR_NAME)
        jdk_tar_path = str(Path(self.install_dir, LINUX_JDK_TAR_NAME))
        self.environment.download_file(LINUX_JDK_URL, jdk_tar_path)
        self.environment.extract_tar(jdk_tar_path, self.install_dir)
        java_home = str(self.find_jdk21_dir(self.install_dir))
        logger.info("OpenJDK 21 installed successfully at %s", java_home)

        # Install Neo4j
        logger.info("Downloading and installing %s...", LINUX_NEO4J_TAR_NAME)
        neo4j_tar_path = str(Path(self.install_dir, LINUX_NEO4J_TAR_NAME))
        self.environment.download_file(LINUX_NEO4J_URL, neo4j_tar_path)
        self.environment.extract_tar(neo4j_tar_path, self.install_dir)
        neo4j_dir = str(Path(self.install_dir) / NEO4J_DIR_NAME)
        logger.info("Neo4j Community Edition installed successfully at %s.", neo4j_dir)

        # Clean up tar files
        Path(jdk_tar_path).unlink(missing_ok=True)
        Path(neo4j_tar_path).unlink(missing_ok=True)

        # Start Neo4j from command line with proper JAVA_HOME
        self.environment.start_neo4j(java_home=java_home, neo4j_dir=neo4j_dir)


class MacosEnvironment(LinuxEnvironment):
    """Environment utilities for macOS installation."""


class MacosInstaller(Installer):
    """Installer for macOS using manual download."""

    def __init__(self, environment: MacosEnvironment | None = None) -> None:
        """Initialize MacosInstaller with default installation directory."""
        super().__init__()
        self.environment = environment or MacosEnvironment()

    def get_run_script_type(self) -> ScriptType:
        """Get the script type for macOS."""
        return ScriptType.BASH

    def check_neo4j_running(self) -> bool:
        """Check if Neo4j service is running on macOS."""
        return self.environment.check_neo4j_running()

    def install_and_start_neo4j(self) -> None:
        """Install and start Neo4j Community Edition on macOS."""
        logger.info("Installing Neo4j Community Edition on macOS...")
        self.ask_install_dir()

        # Determine architecture
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            jdk_url = MACOS_JDK_URL_ARM64
            jdk_tar_name = MACOS_JDK_TAR_NAME_ARM64
        else:
            jdk_url = MACOS_JDK_URL_X64
            jdk_tar_name = MACOS_JDK_TAR_NAME_X64

        # Install OpenJDK 21
        logger.info("Downloading and installing %s...", jdk_tar_name)
        jdk_tar_path = str(Path(self.install_dir, jdk_tar_name))
        self.environment.download_file(jdk_url, jdk_tar_path)
        self.environment.extract_tar(jdk_tar_path, self.install_dir)

        # Find JDK directory
        jdk_root = LinuxInstaller.find_jdk21_dir(self.install_dir)
        java_home = str(jdk_root / "Contents" / "Home")
        logger.info("OpenJDK 21 installed successfully at %s", java_home)

        # Install Neo4j
        logger.info("Downloading and installing %s...", LINUX_NEO4J_TAR_NAME)
        neo4j_tar_path = str(Path(self.install_dir, LINUX_NEO4J_TAR_NAME))
        self.environment.download_file(LINUX_NEO4J_URL, neo4j_tar_path)
        self.environment.extract_tar(neo4j_tar_path, self.install_dir)
        neo4j_dir = str(Path(self.install_dir) / NEO4J_DIR_NAME)
        logger.info("Neo4j Community Edition installed successfully at %s.", neo4j_dir)

        # Clean up tar files
        Path(jdk_tar_path).unlink(missing_ok=True)
        Path(neo4j_tar_path).unlink(missing_ok=True)

        # Start Neo4j from command line with proper JAVA_HOME
        self.environment.start_neo4j(java_home=java_home, neo4j_dir=neo4j_dir)


class WindowsEnvironment:
    """Environment utilities for Windows installation."""

    def download_file(self, url: str, dest: str) -> None:
        """Download file on Windows."""
        urllib.request.urlretrieve(url, dest)

    def extract_zip(self, zip_path: str, extract_to: str) -> None:
        """Extract zip file on Windows."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    def start_neo4j_service(self, install_dir: str) -> None:
        """Install and start Neo4j service on Windows."""
        neo4j_bin_dir = Path(install_dir, NEO4J_DIR_NAME, "bin").resolve()
        env = {**os.environ.copy(), **self.get_neo4j_env(install_dir=install_dir)}
        logger.info("Installing Neo4j service on %s...", install_dir)
        # Initialize Neo4j (required in 5.x before service start)
        subprocess.run(
            [
                str(neo4j_bin_dir / "neo4j-admin.bat"),
                "dbms",
                "set-initial-password",
                DEFAULT_NEO4J_PASSWORD,
            ],
            env={**env.copy(), "NEO4J_AUTH": "neo4j/neo4j"},
            check=True,
        )

        # Install service
        subprocess.run(
            [str(neo4j_bin_dir / "neo4j.bat"), "windows-service", "install"],
            env=env,
            check=True,
        )

        # Start service
        subprocess.run(
            [str(neo4j_bin_dir / "neo4j.bat"), "start"],
            env=env,
            check=True,
        )

    @staticmethod
    def get_neo4j_env(install_dir: str) -> dict[str, str]:
        """Get environment variables for Neo4j on Windows."""
        install_path = Path(install_dir)

        # 1. Dynamically find the JDK directory starting with 'jdk-'
        # next() grabs the first match; returns None if nothing is found
        jdk_path = next(install_path.glob("jdk-*"), None)

        if jdk_path is None:
            logger.error("No directory starting with 'jdk-' found in %s", install_dir)
            # You may want to raise an error here or handle it gracefully
            raise FileNotFoundError(f"Could not find a JDK directory in {install_dir}")

        java_home = str(jdk_path.resolve())
        logger.info("Java home: %s", java_home)

        # 2. Handle Neo4j directory (Assuming NEO4J_DIR_NAME is still defined or similar logic is needed)
        neo4j_home = str(Path(install_dir, NEO4J_DIR_NAME).resolve())
        logger.info("Neo4j home: %s", neo4j_home)

        return {
            "JAVA_HOME": java_home,
            "NEO4J_HOME": neo4j_home,
        }

    def check_neo4j_running(self) -> bool:
        """Check if Neo4j service is running on Windows."""
        try:
            service = psutil.win_service_get(NEO4J_WINDOWS_SERVICE_NAME)  # ty: ignore[possibly-missing-attribute]
            service_info = service.as_dict()
        except Exception as e:
            logger.debug("Failed to check Neo4j status: %s", e)
            return False
        else:
            if service_info["status"] != "running":
                logger.warning(
                    "Neo4j service is installed but not running. "
                    "Please start the service before running MemMachine."
                )
                return False
            return True


class WindowsInstaller(Installer):
    """Installer for Windows using PowerShell."""

    def __init__(self, environment: WindowsEnvironment | None = None) -> None:
        """Initialize WindowsInstaller with default installation directory."""
        super().__init__()
        self.install_dir = os.environ.get("LOCALAPPDATA", "")
        self.install_dir = str(Path(self.install_dir, "MemMachine", "Neo4j"))
        self.environment = environment or WindowsEnvironment()

    def get_run_script_type(self) -> ScriptType:
        """Get the script type for Windows."""
        return ScriptType.POWERSHELL

    def check_neo4j_running(self) -> bool:
        """Check if Neo4j service is running on Windows."""
        return self.environment.check_neo4j_running()

    def install_and_start_neo4j(self) -> None:
        """Install and start Neo4j Community Edition on Windows."""
        logger.info("Installing Neo4j Community Edition...")
        self.ask_install_dir()
        logger.info("Downloading and installing OpenJDK 21...")
        jdk_zip_path = str(Path(self.install_dir, WINDOWS_JDK_ZIP_NAME))
        self.environment.download_file(WINDOWS_JDK_URL, jdk_zip_path)
        self.environment.extract_zip(jdk_zip_path, self.install_dir)
        logger.info("OpenJDK 21 installed successfully.")
        logger.info("Downloading and installing Neo4j Community Edition 2025.09.0...")
        neo4j_zip_path = str(Path(self.install_dir, WINDOWS_NEO4J_ZIP_NAME))
        self.environment.download_file(WINDOWS_NEO4J_URL, neo4j_zip_path)
        self.environment.extract_zip(neo4j_zip_path, self.install_dir)
        logger.info("Neo4j Community Edition installed successfully.")
        # delete zip files
        Path(jdk_zip_path).unlink(missing_ok=True)
        Path(neo4j_zip_path).unlink(missing_ok=True)
        # install and start neo4j service
        logger.info("Starting Neo4j service...")
        self.environment.start_neo4j_service(self.install_dir)
        logger.info("Neo4j service started.")


def install_memmachine() -> None:
    """Install and configure MemMachine based on the operating system."""
    system = platform.system()
    if system == "Windows":
        WindowsInstaller().install()
    elif system == "Darwin":
        MacosInstaller().install()
    elif system == "Linux":
        LinuxInstaller().install()
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")


def main() -> None:
    """Execute the MemMachine installation and configuration script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    try:
        install_memmachine()
        setup_nltk()
        logger.info("MemMachine setup complete.")
        logger.info("Use 'memmachine-server' to start the API server.")
    except KeyboardInterrupt:
        logger.warning("Configuration cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except RuntimeError:
        logger.exception("Configuration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

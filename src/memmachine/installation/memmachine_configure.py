import importlib.resources
import logging
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from abc import ABC, abstractmethod

import psutil

from memmachine.installation.configuration_wizard import (
    ConfigurationWizard,
    ConfigurationWizardArgs,
)

JDK_URL = "https://download.oracle.com/java/21/latest/jdk-21_windows-x64_bin.zip"
NEO4J_URL = "https://dist.neo4j.org/neo4j-community-2025.09.0-windows.zip"
JDK_ZIP_NAME = "jdk-21_windows-x64_bin.zip"
NEO4J_ZIP_NAME = "neo4j-community-2025.09.0-windows.zip"
NEO4J_WINDOWS_SERVICE_NAME = "neo4j"

NEO4J_GPG_KEY_URL = "https://debian.neo4j.com/neotechnology.gpg.key"

NEO4J_GPG_KEY_PATH_DEB = "/etc/apt/keyrings/neotechnology.gpg"
NEO4J_DEB_REPO_ENTRY = "deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest"
NEO4J_DEB_SOURCE_LIST_PATH = "/etc/apt/sources.list.d/neo4j.list"

NEO4J_YUM_REPO_FILE_PATH = "/etc/yum.repos.d/neo4j.repo"
NEO4J_YUM_REPO_ENTRY = """[neo4j]
name=Neo4j RPM Repository
baseurl=https://yum.neo4j.com/stable/latest
enabled=1
gpgcheck=1
EOF"""

JDK_DIR_NAME = "jdk-21.0.9"
NEO4J_DIR_NAME = "neo4j-community-2025.09.0"


class Installer(ABC):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

        # Maybe we can let the user decide these values later
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "neo4j"

    @abstractmethod
    def install_and_start_neo4j(self):
        pass

    @abstractmethod
    def check_neo4j_running(self) -> bool:
        pass

    @abstractmethod
    def get_run_script_type(self) -> str:
        pass

    def install(self):
        neo4j_started_by_installer = False
        if not self.check_neo4j_running():
            choice = (
                input(
                    "Neo4j is not running. Do you want to install and start Neo4j? (y/n): "
                )
                .strip()
                .lower()
            )
            if choice != "y":
                raise Exception(
                    "Neo4j installation is required to proceed. Exiting installation."
                )
            self.install_and_start_neo4j()
            neo4j_started_by_installer = True
        else:
            self.logger.info("Neo4j is already running.")

        wizard_args = ConfigurationWizardArgs(
            neo4j_provided=neo4j_started_by_installer,
            neo4j_uri=self.neo4j_uri,
            neo4j_username=self.neo4j_username,
            neo4j_password=self.neo4j_password,
            config_sources={
                "CPU": importlib.resources.files("memmachine").joinpath(
                    "sample_configs/episodic_memory_config.cpu.sample"
                ),
                "GPU": importlib.resources.files("memmachine").joinpath(
                    "sample_configs/episodic_memory_config.gpu.sample"
                ),
            },
            destination=os.path.join(os.path.expanduser("~"), ".config", "memmachine"),
            run_script_type=self.get_run_script_type(),
        )
        wizard = ConfigurationWizard(
            wizard_args,
            logger=self.logger,
        )
        wizard.run_wizard()

        self.logger.info("MemMachine installation and configuration completed.")
        self.logger.info(
            "Please run memmachine-nltk-setup and memmachine-sync-profile-schema to complete the setup."
        )


class LinuxEnvironment:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def get_package_manager(self) -> str:
        if shutil.which("apt"):
            return "apt"
        elif shutil.which("dnf"):
            return "dnf"
        elif shutil.which("yum"):
            return "yum"
        else:
            raise Exception(
                "Unsupported Linux distribution: No known package manager found. Supported package managers are apt, dnf, and yum."
            )

    def download_content(self, url: str) -> bytes:
        with urllib.request.urlopen(url) as response:
            return response.read()

    def save_gpg_key(self, gpg_key_data: str, path: str):
        subprocess.run(
            ["sudo", "gpg", "--dearmor", "-o", path],
            input=gpg_key_data,
            text=True,
            check=True,
        )

    def apt_update(self):
        subprocess.run(["sudo", "apt-get", "update"], check=True)

    def import_rpm_gpg_key(self, url: str):
        subprocess.run(
            ["sudo", "rpm", "--import", url],
            check=True,
        )

    def apt_install(self, package: str):
        subprocess.run(["sudo", "apt-get", "install", "-y", package], check=True)

    def yum_install(self, package_manager: str, package: str):
        subprocess.run([package_manager, "install", "-y", package], check=True)

    def systemctl_enable_service(self, service_name: str):
        subprocess.run(["sudo", "systemctl", "enable", service_name], check=True)
        subprocess.run(["sudo", "systemctl", "start", service_name], check=True)

    def write_file_with_sudo(self, path: str, content: str):
        subprocess.run(
            ["sudo", "tee", path],
            input=content,
            text=True,
            check=True,
        )

    def check_neo4j_running(self) -> bool:
        try:
            status_result = subprocess.run(
                ["systemctl", "status", "neo4j"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            if status_result.returncode == 4:
                return False
        except FileNotFoundError:
            self.logger.error(
                "systemctl not found. Cannot check Neo4j service status or install Neo4j."
            )
            return False

        active_result = subprocess.run(
            ["systemctl", "is-active", "neo4j"], capture_output=True, text=True
        )
        if active_result.stdout.strip() != "active":
            self.logger.warning(
                "Neo4j service is installed but not running. Please start the service before running MemMachine."
            )
        return True


class LinuxInstaller(Installer):
    def __init__(
        self, logger: logging.Logger, environment: LinuxEnvironment | None = None
    ):
        super().__init__(logger)
        self.environment = environment or LinuxEnvironment(logger)
        self.package_manager = self.environment.get_package_manager()

    def install_with_apt(self):
        self.logger.info("Adding Neo4j repository and installing Neo4j...")
        gpg_key_data = self.environment.download_content(NEO4J_GPG_KEY_URL)
        self.logger.info(f"Saving GPG key to {NEO4J_GPG_KEY_PATH_DEB}...")
        self.environment.save_gpg_key(
            gpg_key_data.decode("ascii"), NEO4J_GPG_KEY_PATH_DEB
        )
        self.logger.info(f"Adding Neo4j repository to {NEO4J_DEB_SOURCE_LIST_PATH}...")
        self.environment.write_file_with_sudo(
            NEO4J_DEB_SOURCE_LIST_PATH, NEO4J_DEB_REPO_ENTRY
        )
        self.logger.info("Updating package lists...")
        self.environment.apt_update()
        self.logger.info("Installing Neo4j package...")
        self.environment.apt_install("neo4j")

    def install_with_dnf_or_yum(self, package_manager: str):
        self.logger.info("Adding Neo4j repository and installing Neo4j...")
        self.environment.import_rpm_gpg_key(NEO4J_GPG_KEY_URL)
        self.logger.info(f"Adding Neo4j repository to {NEO4J_YUM_REPO_FILE_PATH}...")
        self.environment.write_file_with_sudo(
            NEO4J_YUM_REPO_FILE_PATH, NEO4J_YUM_REPO_ENTRY
        )
        self.logger.info("Installing Neo4j package...")
        self.environment.yum_install(package_manager, "neo4j")

    def get_run_script_type(self):
        return "bash"

    def check_neo4j_running(self) -> bool:
        return self.environment.check_neo4j_running()

    def install_and_start_neo4j(self):
        self.logger.info(
            f"Installing Neo4j Community Edition with {self.package_manager}..."
        )
        match self.package_manager:
            case "apt":
                self.install_with_apt()
            case "dnf" | "yum":
                self.install_with_dnf_or_yum(self.package_manager)
            case _:
                raise Exception(f"Unsupported package manager: {self.package_manager}")
        self.logger.info("Enabling and starting Neo4j service...")
        self.environment.systemctl_enable_service("neo4j")
        self.logger.info("Neo4j service installed successfully.")


class MacosEnvironment:
    def install_and_start_neo4j(self):
        subprocess.run(["brew", "install", "neo4j"], check=True)
        subprocess.run(["brew", "services", "start", "neo4j"], check=True)

    def check_neo4j_running(self) -> bool:
        result = subprocess.run(
            ["brew", "list", "--versions", "neo4j"], capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() != ""


class MacosInstaller(Installer):
    def __init__(
        self, logger: logging.Logger, environment: MacosEnvironment | None = None
    ):
        super().__init__(logger)
        self.environment = environment or MacosEnvironment()

    def get_run_script_type(self):
        return "bash"

    def check_neo4j_running(self) -> bool:
        return self.environment.check_neo4j_running()

    def install_and_start_neo4j(self):
        self.environment.install_and_start_neo4j()


class WindowsEnvironment:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def download_file(self, url: str, dest: str):
        curl_path = shutil.which("curl.exe")
        if not curl_path:
            urllib.request.urlretrieve(url, dest)
            return
        subprocess.run(["curl.exe", "-L", "-o", dest, url], check=True)

    def extract_zip(self, zip_path: str, extract_to: str):
        tar_path = shutil.which("tar.exe")
        if not tar_path:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            return
        subprocess.run(["tar", "-xf", zip_path, "-C", extract_to], check=True)

    def start_neo4j_service(self, install_dir: str):
        subprocess.run(
            [
                "powershell.exe",
                "-File",
                os.path.join(install_dir, NEO4J_DIR_NAME, "bin", "neo4j.ps1"),
                "windows-service",
                "install",
            ],
            env={**os.environ.copy(), **self.get_neo4j_env(install_dir=install_dir)},
            check=True,
        )
        subprocess.run(
            [
                "powershell.exe",
                "-File",
                os.path.join(install_dir, NEO4J_DIR_NAME, "bin", "neo4j.ps1"),
                "start",
            ],
            env={**os.environ.copy(), **self.get_neo4j_env(install_dir=install_dir)},
            check=True,
        )

    def get_neo4j_env(self, install_dir: str) -> dict[str, str]:
        return {
            "JAVA_HOME": os.path.join(install_dir, JDK_DIR_NAME),
            "NEO4J_HOME": os.path.join(install_dir, NEO4J_DIR_NAME),
        }

    def check_neo4j_running(self) -> bool:
        try:
            service = psutil.win_service_get(NEO4J_WINDOWS_SERVICE_NAME)
            service_info = service.as_dict()
            if service_info["status"] != "running":
                self.logger.warning(
                    "Neo4j service is installed but not running. Please start the service before running MemMachine."
                )
            return True
        except Exception:
            return False


class WindowsInstaller(Installer):
    def __init__(
        self, logger: logging.Logger, environment: WindowsEnvironment | None = None
    ):
        super().__init__(logger)
        self.install_dir = os.environ.get("LOCALAPPDATA", "")
        if self.install_dir:
            self.install_dir = os.path.join(self.install_dir, "MemMachine", "Neo4j")
        self.environment = environment or WindowsEnvironment(logger)

    def get_run_script_type(self):
        return "powershell"

    def check_neo4j_running(self) -> bool:
        return self.environment.check_neo4j_running()

    def ask_install_dir(self):
        while True:
            install_dir = input(
                f"Enter Neo4j installation directory [{self.install_dir}]: "
            ).strip()
            if install_dir:
                self.install_dir = install_dir

            if self.install_dir:
                break
            self.logger.error("Installation directory cannot be empty.")
            continue
        if not os.path.exists(self.install_dir):
            self.logger.info(
                f"Installation directory {self.install_dir} does not exist. Creating..."
            )
        else:
            choice = (
                input(
                    f"Installation directory {self.install_dir} already exists. Would you like to remove it? (y/n): "
                )
                .strip()
                .lower()
            )
            if choice != "y":
                raise Exception(
                    "Existing installation directory must be removed to proceed. Exiting installation."
                )
            self.logger.info(
                f"Removing existing installation directory {self.install_dir}..."
            )
            shutil.rmtree(self.install_dir)
        os.makedirs(self.install_dir, exist_ok=True)

    def install_and_start_neo4j(self):
        self.logger.info("Installing Neo4j Community Edition...")
        self.ask_install_dir()
        self.logger.info("Downloading and installing OpenJDK 21...")
        jdk_zip_path = os.path.join(self.install_dir, JDK_ZIP_NAME)
        self.environment.download_file(JDK_URL, jdk_zip_path)
        self.environment.extract_zip(jdk_zip_path, self.install_dir)
        self.logger.info("OpenJDK 21 installed successfully.")
        self.logger.info(
            "Downloading and installing Neo4j Community Edition 2025.09.0..."
        )
        neo4j_zip_path = os.path.join(self.install_dir, NEO4J_ZIP_NAME)
        self.environment.download_file(NEO4J_URL, neo4j_zip_path)
        self.environment.extract_zip(neo4j_zip_path, self.install_dir)
        self.logger.info("Neo4j Community Edition installed successfully.")
        # delete zip files
        os.remove(jdk_zip_path)
        os.remove(neo4j_zip_path)
        # install and start neo4j service
        self.logger.info("Starting Neo4j service...")
        self.environment.start_neo4j_service(self.install_dir)
        self.logger.info("Neo4j service started.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("MemMachineInstaller")
    try:
        system = platform.system()
        if system == "Windows":
            WindowsInstaller(logger).install()
        elif system == "Darwin":
            MacosInstaller(logger).install()
        elif system == "Linux":
            LinuxInstaller(logger).install()
        else:
            raise Exception(f"Unsupported operating system: {system}")
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        os.exit(1)


if __name__ == "__main__":
    main()

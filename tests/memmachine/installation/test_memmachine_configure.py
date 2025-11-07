import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from regex import P

from memmachine.installation.configuration_wizard import ConfigurationWizardArgs
from memmachine.installation.memmachine_configure import (
    JDK_URL,
    JDK_ZIP_NAME,
    NEO4J_DEB_REPO_ENTRY,
    NEO4J_DEB_SOURCE_LIST_PATH,
    NEO4J_GPG_KEY_PATH_DEB,
    NEO4J_GPG_KEY_URL,
    NEO4J_URL,
    NEO4J_YUM_REPO_ENTRY,
    NEO4J_YUM_REPO_FILE_PATH,
    NEO4J_ZIP_NAME,
    ConfigurationWizard,
    LinuxEnvironment,
    LinuxInstaller,
    MacosEnvironment,
    MacosInstaller,
    WindowsEnvironment,
    WindowsInstaller,
)

MOCK_INSTALL_DIR = "C:\\Users\\TestUser\\MemMachine"
MOCK_LOCALDATA_DIR = "C:\\Users\\TestUser\\AppData\\Local"
MOCK_GPG_KEY_CONTENT = "mocked-gpg-key-content"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("MemMachineInstaller")


def mock_wizard_run(self):
    os.makedirs(self.args.destination, exist_ok=True)
    Path(os.path.join(self.args.destination, "configuration.yml")).touch()
    if self.args.run_script_type == "bash":
        Path(os.path.join(self.args.destination, "run_script.sh")).touch()
    else:
        Path(os.path.join(self.args.destination, "run_script.ps1")).touch()


def mock_wizard_init(self, args: ConfigurationWizardArgs, logger: logging.Logger):
    self.args = args


@pytest.fixture()
def mock_wizard():
    with patch.object(ConfigurationWizard, "__init__", mock_wizard_init):
        with patch.object(ConfigurationWizard, "run_wizard", mock_wizard_run):
            yield


class MockWindowsEnvironment(WindowsEnvironment):
    def __init__(self, logger):
        super().__init__(logger)
        self.expected_install_dir = MOCK_INSTALL_DIR
        self.openjdk_zip_downloaded = False
        self.neo4j_zip_downloaded = False
        self.openjdk_extracted = False
        self.neo4j_extracted = False
        self.neo4j_installed = False
        self.neo4j_uninstalled = False
        self.neo4j_preinstalled = False

    def download_file(self, url: str, dest: str):
        if url == JDK_URL:
            assert dest == os.path.join(self.expected_install_dir, JDK_ZIP_NAME)
            Path(dest).touch()  # Create an empty file to simulate download
            self.openjdk_zip_downloaded = True
        elif url == NEO4J_URL:
            assert dest == os.path.join(self.expected_install_dir, NEO4J_ZIP_NAME)
            Path(dest).touch()  # Create an empty file to simulate download
            self.neo4j_zip_downloaded = True
        else:
            raise ValueError("Unexpected URL")

    def extract_zip(self, zip_path: str, extract_to: str):
        assert extract_to == self.expected_install_dir
        if zip_path == os.path.join(self.expected_install_dir, JDK_ZIP_NAME):
            assert self.openjdk_zip_downloaded
            self.openjdk_extracted = True
        elif zip_path == os.path.join(self.expected_install_dir, NEO4J_ZIP_NAME):
            assert self.neo4j_zip_downloaded
            self.neo4j_extracted = True
        else:
            raise ValueError("Unexpected zip path")

    def start_neo4j_service(self, install_dir: str):
        assert install_dir == self.expected_install_dir
        assert self.neo4j_extracted
        assert self.openjdk_extracted
        self.neo4j_installed = True

    def check_neo4j_running(self) -> bool:
        return self.neo4j_preinstalled


@patch("builtins.input")
def test_install_in_windows(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
        MOCK_INSTALL_DIR,  # Install directory
    ]
    installer = WindowsInstaller(logger, MockWindowsEnvironment(logger))
    installer.install()
    assert installer.environment.neo4j_installed
    assert os.path.exists(MOCK_INSTALL_DIR)
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME))
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


@patch("builtins.input")
def test_remove_old_dir_in_windows(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
        MOCK_INSTALL_DIR,  # Install directory
        "y",  # Confirm removal of old directory
    ]
    os.makedirs(MOCK_INSTALL_DIR, exist_ok=True)
    Path(os.path.join(MOCK_INSTALL_DIR, "old_file.txt")).touch()
    installer = WindowsInstaller(logger, MockWindowsEnvironment(logger))
    installer.install()
    assert installer.environment.neo4j_installed
    assert os.path.exists(MOCK_INSTALL_DIR)
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, "old_file.txt"))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME))
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


@patch("builtins.input")
def test_not_remove_old_dir_in_windows(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
        MOCK_INSTALL_DIR,  # Install directory
        "n",  # Do not remove old directory
    ]
    os.makedirs(MOCK_INSTALL_DIR, exist_ok=True)
    Path(os.path.join(MOCK_INSTALL_DIR, "old_file.txt")).touch()
    installer = WindowsInstaller(logger, MockWindowsEnvironment(logger))
    with pytest.raises(Exception) as excinfo:
        installer.install()
    assert (
        str(excinfo.value)
        == "Existing installation directory must be removed to proceed. Exiting installation."
    )
    assert os.path.exists(os.path.join(MOCK_INSTALL_DIR, "old_file.txt"))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME))
    assert not os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert not os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


@patch("builtins.input")
def test_install_in_windows_default_dir(mock_input, fs, monkeypatch, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
        "",  # Use default install directory
    ]
    monkeypatch.setenv("LOCALAPPDATA", MOCK_LOCALDATA_DIR)
    environment = MockWindowsEnvironment(logger)
    environment.expected_install_dir = os.path.join(
        MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j"
    )
    installer = WindowsInstaller(logger, environment)
    installer.install()
    assert installer.environment.neo4j_installed
    assert os.path.exists(os.path.join(MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j"))
    assert not os.path.exists(
        os.path.join(MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j", JDK_ZIP_NAME)
    )
    assert not os.path.exists(
        os.path.join(MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j", NEO4J_ZIP_NAME)
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


@patch("builtins.input")
def test_install_in_windows_no_default_dir(mock_input, fs, monkeypatch, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
        "",  # No default install directory set
        "",
        "",  # we will ask until a valid directory is provided
        MOCK_INSTALL_DIR,  # Provide install directory
    ]
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    environment = MockWindowsEnvironment(logger)
    environment.expected_install_dir = MOCK_INSTALL_DIR
    installer = WindowsInstaller(logger, environment)
    installer.install()
    assert installer.environment.neo4j_installed
    assert os.path.exists(MOCK_INSTALL_DIR)
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME))
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


@patch("builtins.input")
def test_cancel_install_in_windows(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "n",  # Cancel installation
    ]
    installer = WindowsInstaller(logger, MockWindowsEnvironment(logger))
    with pytest.raises(Exception) as excinfo:
        installer.install()
    assert (
        str(excinfo.value)
        == "Neo4j installation is required to proceed. Exiting installation."
    )
    assert not installer.environment.neo4j_installed
    assert not os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert not os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


def test_install_in_windows_neo4j_preinstalled(fs, mock_wizard):
    installer = WindowsInstaller(logger, MockWindowsEnvironment(logger))
    installer.environment.neo4j_preinstalled = True
    installer.install()
    assert not installer.environment.neo4j_installed
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.ps1")
    )


class MockMacOSEnvironment(MacosEnvironment):
    def __init__(self):
        super().__init__()
        self.neo4j_installed = False
        self.neo4j_preinstalled = False

    def install_and_start_neo4j(self):
        self.neo4j_installed = True

    def check_neo4j_running(self) -> bool:
        return self.neo4j_preinstalled


@patch("builtins.input")
def test_install_in_macos(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    installer = MacosInstaller(logger, MockMacOSEnvironment())
    installer.install()
    assert installer.environment.neo4j_installed
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.sh")
    )


def test_install_in_macos_neo4j_preinstalled(fs, mock_wizard):
    environment = MockMacOSEnvironment()
    environment.neo4j_preinstalled = True
    installer = MacosInstaller(logger, environment)
    installer.install()
    assert not installer.environment.neo4j_installed
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.sh")
    )


class MockLinuxEnvironment(LinuxEnvironment):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.package_manager = ""
        self.deb_gpg_key_saved = False
        self.apt_updated = False
        self.rpm_gpg_key_imported = False
        self.apt_installed = False
        self.yum_installed = False
        self.neo4j_started = False
        self.neo4j_preinstalled = False

    def get_package_manager(self) -> str:
        return self.package_manager

    def download_content(self, url: str) -> bytes:
        assert url == NEO4J_GPG_KEY_URL
        return MOCK_GPG_KEY_CONTENT.encode("ascii")

    def save_gpg_key(self, gpg_key_data: str, path: str):
        assert gpg_key_data == MOCK_GPG_KEY_CONTENT
        assert path == NEO4J_GPG_KEY_PATH_DEB
        self.deb_gpg_key_saved = True

    def apt_update(self):
        assert self.deb_gpg_key_saved
        self.apt_updated = True

    def import_rpm_gpg_key(self, url: str):
        assert url == NEO4J_GPG_KEY_URL
        self.rpm_gpg_key_imported = True

    def apt_install(self, package: str):
        assert package == "neo4j"
        assert self.apt_updated
        self.apt_installed = True

    def yum_install(self, package_manager: str, package: str):
        assert self.rpm_gpg_key_imported
        assert package_manager == self.package_manager
        assert package == "neo4j"
        self.yum_installed = True

    def systemctl_enable_service(self, service_name: str):
        assert service_name == "neo4j"
        if self.package_manager == "apt":
            assert self.apt_installed
        else:
            assert self.yum_installed
        self.neo4j_started = True

    def write_file_with_sudo(self, path: str, content: str):
        assert path in [NEO4J_YUM_REPO_FILE_PATH, NEO4J_DEB_SOURCE_LIST_PATH]
        if path == NEO4J_YUM_REPO_FILE_PATH:
            assert content == NEO4J_YUM_REPO_ENTRY
        else:
            assert content == NEO4J_DEB_REPO_ENTRY
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def check_neo4j_running(self) -> bool:
        return self.neo4j_preinstalled


@patch("builtins.input")
def test_install_in_linux_apt(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    environment = MockLinuxEnvironment(logger)
    environment.package_manager = "apt"
    installer = LinuxInstaller(logger, environment)
    installer.install()
    assert environment.apt_installed
    assert environment.neo4j_started
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.sh")
    )


@patch("builtins.input")
def test_install_in_linux_yum(mock_input, fs, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    environment = MockLinuxEnvironment(logger)
    environment.package_manager = "yum"
    installer = LinuxInstaller(logger, environment)
    installer.install()
    assert environment.yum_installed
    assert environment.neo4j_started
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.sh")
    )


def test_install_in_linux_neo4j_preinstalled(fs, mock_wizard):
    environment = MockLinuxEnvironment(logger)
    environment.package_manager = "apt"
    environment.neo4j_preinstalled = True
    installer = LinuxInstaller(logger, environment)
    installer.install()
    assert not environment.apt_installed
    assert not environment.yum_installed
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "configuration.yml")
    )
    assert os.path.exists(
        os.path.join(os.path.expanduser("~/.config/memmachine"), "run_script.sh")
    )

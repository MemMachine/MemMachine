import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from memmachine.installation.memmachine_configure import (
    WINDOWS_JDK_URL,
    WINDOWS_JDK_ZIP_NAME,
    WINDOWS_NEO4J_URL,
    WINDOWS_NEO4J_ZIP_NAME,
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
    Path(self.args.destination).mkdir(parents=True, exist_ok=True)
    Path(self.args.destination, "cfg.yml").touch()


def mock_wizard_init(self, args: ConfigurationWizard.Params):
    self.args = args


@pytest.fixture
def mock_wizard():
    with (
        patch.object(ConfigurationWizard, "__init__", mock_wizard_init),
        patch.object(ConfigurationWizard, "run_wizard", mock_wizard_run),
    ):
        yield


class MockWindowsEnvironment(WindowsEnvironment):
    def __init__(self):
        super().__init__()
        self.expected_install_dir = MOCK_INSTALL_DIR
        self.openjdk_zip_downloaded = False
        self.neo4j_zip_downloaded = False
        self.openjdk_extracted = False
        self.neo4j_extracted = False
        self.neo4j_installed = False
        self.neo4j_uninstalled = False
        self.neo4j_preinstalled = False

    def download_file(self, url: str, dest: str):
        if url == WINDOWS_JDK_URL:
            assert dest == str(Path(self.expected_install_dir) / WINDOWS_JDK_ZIP_NAME)
            Path(dest).touch()  # Create an empty file to simulate download
            self.openjdk_zip_downloaded = True
        elif url == WINDOWS_NEO4J_URL:
            assert dest == str(Path(self.expected_install_dir) / WINDOWS_NEO4J_ZIP_NAME)
            Path(dest).touch()  # Create an empty file to simulate download
            self.neo4j_zip_downloaded = True
        else:
            raise ValueError("Unexpected URL")

    def extract_zip(self, zip_path: str, extract_to: str):
        assert extract_to == self.expected_install_dir
        if zip_path == str(Path(self.expected_install_dir) / WINDOWS_JDK_ZIP_NAME):
            assert self.openjdk_zip_downloaded
            self.openjdk_extracted = True
        elif zip_path == str(Path(self.expected_install_dir) / WINDOWS_NEO4J_ZIP_NAME):
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
def test_install_in_windows(mock_input, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    installer = WindowsInstaller(MockWindowsEnvironment())
    installer.install_dir = MOCK_INSTALL_DIR
    installer.install()
    assert installer.environment.neo4j_installed
    assert Path(MOCK_INSTALL_DIR).exists()
    assert not (Path(MOCK_INSTALL_DIR) / WINDOWS_JDK_ZIP_NAME).exists()
    assert not Path(MOCK_INSTALL_DIR, WINDOWS_NEO4J_ZIP_NAME).exists()
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


@patch("builtins.input")
def test_install_in_windows_default_dir(mock_input, monkeypatch, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
        "",  # Use default install directory
    ]
    monkeypatch.setenv("LOCALAPPDATA", MOCK_LOCALDATA_DIR)
    environment = MockWindowsEnvironment()
    neo4j_path = Path(MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j")
    environment.expected_install_dir = str(neo4j_path)
    installer = WindowsInstaller(environment)
    installer.install()
    assert installer.environment.neo4j_installed
    assert neo4j_path.exists()
    assert not Path(
        MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j", WINDOWS_JDK_ZIP_NAME
    ).exists()
    assert not Path(
        MOCK_LOCALDATA_DIR, "MemMachine", "Neo4j", WINDOWS_NEO4J_ZIP_NAME
    ).exists()
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


@patch("builtins.input")
def test_use_custom_neo4j(mock_input, mock_wizard):
    mock_input.side_effect = [
        "n",  # do not install neo4j
    ]
    installer = WindowsInstaller(MockWindowsEnvironment())
    installer.install()
    assert not installer.environment.neo4j_installed


def test_install_in_windows_neo4j_preinstalled(mock_wizard):
    installer = WindowsInstaller(MockWindowsEnvironment())
    installer.environment.neo4j_preinstalled = True
    installer.install()
    assert not installer.environment.neo4j_installed
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


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
def test_install_in_macos(mock_input, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    installer = MacosInstaller(MockMacOSEnvironment())
    installer.install()
    assert installer.environment.neo4j_installed
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


def test_install_in_macos_neo4j_preinstalled(mock_wizard):
    environment = MockMacOSEnvironment()
    environment.neo4j_preinstalled = True
    installer = MacosInstaller(environment)
    installer.install()
    assert not installer.environment.neo4j_installed
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


class MockLinuxEnvironment(LinuxEnvironment):
    def __init__(self):
        self.neo4j_started = False
        self.downloaded_files = {}
        self.extracted_files = {}

    def download_file(self, url: str, dest: str) -> None:
        self.downloaded_files[url] = dest

    def extract_tar(self, zip_path: str, extract_to: str) -> None:
        self.extracted_files[zip_path] = extract_to

    def start_neo4j(self, java_home: str, neo4j_dir: str) -> None:
        self.neo4j_started = True

    def check_neo4j_running(self) -> bool:
        return False


@patch("builtins.input")
def test_install_in_linux(mock_input, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    environment = MockLinuxEnvironment()
    installer = LinuxInstaller(environment)
    installer.install()
    assert environment.neo4j_started

import io
import logging
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memmachine_server.installation.memmachine_configure import (
    LINUX_NEO4J_URL,
    MACOS_JDK_URL_ARM64,
    MACOS_JDK_URL_X64,
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
    _safe_extract_zip,
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
    environment = MockWindowsEnvironment()
    installer = WindowsInstaller(environment)
    installer.install_dir = MOCK_INSTALL_DIR
    installer.install()
    assert environment.neo4j_installed
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
    assert environment.neo4j_installed
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
    environment = MockWindowsEnvironment()
    installer = WindowsInstaller(environment)
    installer.install()
    assert not environment.neo4j_installed


def test_install_in_windows_neo4j_preinstalled(mock_wizard):
    environment = MockWindowsEnvironment()
    installer = WindowsInstaller(environment)
    environment.neo4j_preinstalled = True
    installer.install()
    assert not environment.neo4j_installed
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


class MockMacOSEnvironment(MacosEnvironment):
    def __init__(self):
        super().__init__()
        self.neo4j_started = False
        self.neo4j_preinstalled = False
        self.downloaded_files = {}
        self.extracted_files = {}

    def download_file(self, url: str, dest: str) -> None:
        self.downloaded_files[url] = dest

    def extract_tar(self, tar_path: str, extract_to: str) -> None:
        self.extracted_files[tar_path] = extract_to

    def start_neo4j(self, java_home: str, neo4j_dir: str) -> None:
        self.neo4j_started = True

    def check_neo4j_running(self) -> bool:
        return self.neo4j_preinstalled


@patch("builtins.input")
@patch("platform.machine")
def test_install_in_macos(mock_machine, mock_input, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    mock_machine.return_value = "arm64"
    environment = MockMacOSEnvironment()
    installer = MacosInstaller(environment)
    installer.install()
    assert environment.neo4j_started
    assert MACOS_JDK_URL_ARM64 in environment.downloaded_files
    assert LINUX_NEO4J_URL in environment.downloaded_files
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


@patch("builtins.input")
@patch("platform.machine")
def test_install_in_macos_x64(mock_machine, mock_input, mock_wizard):
    mock_input.side_effect = [
        "y",  # Confirm installation
    ]
    mock_machine.return_value = "x86_64"
    environment = MockMacOSEnvironment()
    installer = MacosInstaller(environment)
    installer.install()
    assert environment.neo4j_started
    assert MACOS_JDK_URL_X64 in environment.downloaded_files
    assert LINUX_NEO4J_URL in environment.downloaded_files
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


def test_install_in_macos_neo4j_preinstalled(mock_wizard):
    environment = MockMacOSEnvironment()
    environment.neo4j_preinstalled = True
    installer = MacosInstaller(environment)
    installer.install()
    assert not environment.neo4j_started
    assert Path("~/.config/memmachine/cfg.yml").expanduser().exists()


class MockLinuxEnvironment(LinuxEnvironment):
    def __init__(self):
        self.neo4j_started = False
        self.downloaded_files = {}
        self.extracted_files = {}

    def download_file(self, url: str, dest: str) -> None:
        self.downloaded_files[url] = dest

    def extract_tar(self, tar_path: str, extract_to: str) -> None:
        self.extracted_files[tar_path] = extract_to

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


# ---------------------------------------------------------------------------
# Helper: build an in-memory zip and write it to a temp file
# ---------------------------------------------------------------------------


def _make_zip(tmp_path: Path, entries: list[tuple[str, bytes, int]]) -> str:
    """Create a zip file at *tmp_path/test.zip* with the given entries.

    Each entry is a tuple of (filename, data, external_attr).
    ``external_attr`` is placed in the ZipInfo field of the same name so
    tests can inject Unix file-type bits (e.g. symlink mask 0xA000 << 16).
    """
    zip_path = str(tmp_path / "test.zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data, ext_attr in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = ext_attr
            zf.writestr(info, data)
    (tmp_path / "test.zip").write_bytes(buf.getvalue())
    return zip_path


# ---------------------------------------------------------------------------
# _safe_extract_zip - path-traversal tests
# ---------------------------------------------------------------------------


def test_safe_extract_zip_normal(tmp_path: Path) -> None:
    """A clean zip extracts all files to the target directory."""
    extract_to = tmp_path / "out"
    extract_to.mkdir()
    zip_path = _make_zip(
        tmp_path,
        [
            ("subdir/file.txt", b"hello", 0),
            ("top.txt", b"world", 0),
        ],
    )
    _safe_extract_zip(zip_path, str(extract_to))
    assert (extract_to / "subdir" / "file.txt").read_bytes() == b"hello"
    assert (extract_to / "top.txt").read_bytes() == b"world"


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    """An entry with ``../`` in its name must raise ValueError."""
    extract_to = tmp_path / "out"
    extract_to.mkdir()
    zip_path = _make_zip(
        tmp_path,
        [("../evil.txt", b"pwned", 0)],
    )
    with pytest.raises(ValueError, match="would extract outside"):
        _safe_extract_zip(zip_path, str(extract_to))


def test_safe_extract_zip_rejects_absolute_path(tmp_path: Path) -> None:
    """An entry with an absolute path must raise ValueError."""
    extract_to = tmp_path / "out"
    extract_to.mkdir()
    zip_path = _make_zip(
        tmp_path,
        [("/etc/passwd", b"bad", 0)],
    )
    with pytest.raises(ValueError, match="would extract outside"):
        _safe_extract_zip(zip_path, str(extract_to))


def test_safe_extract_zip_rejects_symlink_entry(tmp_path: Path) -> None:
    """An entry whose external_attr marks it as a symlink must raise ValueError."""
    extract_to = tmp_path / "out"
    extract_to.mkdir()
    # Unix symlink type: 0xA000 in the upper 16 bits of external_attr
    symlink_attr = 0xA000 << 16
    zip_path = _make_zip(
        tmp_path,
        [("link", b"../secret", symlink_attr)],
    )
    with pytest.raises(ValueError, match="symlink"):
        _safe_extract_zip(zip_path, str(extract_to))


def test_safe_extract_zip_no_traversal_no_false_positive(tmp_path: Path) -> None:
    """A filename that merely *contains* '..' as a substring is fine."""
    extract_to = tmp_path / "out"
    extract_to.mkdir()
    zip_path = _make_zip(
        tmp_path,
        [("file..name.txt", b"ok", 0)],
    )
    # Should not raise
    _safe_extract_zip(zip_path, str(extract_to))
    assert (extract_to / "file..name.txt").read_bytes() == b"ok"


# ---------------------------------------------------------------------------
# WindowsEnvironment.extract_zip - delegates to _safe_extract_zip
# ---------------------------------------------------------------------------


def test_windows_environment_extract_zip_delegates(tmp_path: Path) -> None:
    """WindowsEnvironment.extract_zip uses _safe_extract_zip and rejects traversal."""
    extract_to = tmp_path / "out"
    extract_to.mkdir()
    zip_path = _make_zip(
        tmp_path,
        [("../escape.txt", b"bad", 0)],
    )
    env = WindowsEnvironment()
    with pytest.raises(ValueError, match="would extract outside"):
        env.extract_zip(zip_path, str(extract_to))

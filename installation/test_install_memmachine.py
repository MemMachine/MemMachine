import os
from unittest.mock import patch

from install_memmachine import (
    BASH_STARTUP_TEMPLATE,
    CONFIGURATION_TEMPLATE,
    JDK_URL,
    JDK_ZIP_NAME,
    MACOS_RUN_SCRIPT_NAME,
    NEO4J_URL,
    NEO4J_ZIP_NAME,
    POWERSHELL_STARTUP_TEMPLATE,
    WINDOWS_RUN_SCRIPT_NAME,
    MacosEnvironment,
    MacosInstaller,
    WindowsEnvironment,
    WindowsInstaller,
)

MOCK_INSTALL_DIR = "/mock/install/dir"
MOCK_OPENAI_API_KEY = "mock_openai_api_key"


def assert_file_content(file_path: str, expected_content: str):
    with open(file_path, "r") as f:
        content = f.read()
    assert content == expected_content


class MockMacosEnvironment(MacosEnvironment):
    def __init__(self):
        super().__init__()
        self._neo4j_installed = False
        self.mem_machine_installed = False
        self.neo4j_install_called = False

    def install_memmachine_in_venv(self, install_dir: str):
        assert install_dir == MOCK_INSTALL_DIR
        assert self._neo4j_installed
        self.mem_machine_installed = True

    def install_neo4j(self):
        self.neo4j_install_called = True
        self._neo4j_installed = True

    def neo4j_installed(self) -> bool:
        return self._neo4j_installed


@patch("builtins.input")
def test_install_in_macos(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        MOCK_OPENAI_API_KEY,  # OpenAI API Key
    ]
    installer = MacosInstaller(MockMacosEnvironment())
    installer.install()
    assert installer.environment.mem_machine_installed
    assert os.path.exists(MOCK_INSTALL_DIR)
    assert os.path.exists(os.path.join(MOCK_INSTALL_DIR, "configuration.yml"))
    assert_file_content(
        os.path.join(MOCK_INSTALL_DIR, "configuration.yml"),
        CONFIGURATION_TEMPLATE.format(OPENAI_KEY=MOCK_OPENAI_API_KEY),
    )
    assert os.path.exists(os.path.join(MOCK_INSTALL_DIR, MACOS_RUN_SCRIPT_NAME))
    assert_file_content(
        os.path.join(MOCK_INSTALL_DIR, MACOS_RUN_SCRIPT_NAME),
        BASH_STARTUP_TEMPLATE.format(OPENAI_KEY=MOCK_OPENAI_API_KEY),
    )


@patch("builtins.input")
def test_install_in_macos_neo4j_already_installed(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        MOCK_OPENAI_API_KEY,  # OpenAI API Key
    ]
    mock_env = MockMacosEnvironment()
    mock_env._neo4j_installed = True  # Simulate Neo4j already installed
    installer = MacosInstaller(mock_env)
    installer.install()
    assert installer.environment.mem_machine_installed
    assert not installer.environment.neo4j_install_called
    assert os.path.exists(MOCK_INSTALL_DIR)


@patch("builtins.input")
def test_path_already_exists(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        "y",  # Confirm uninstall
        "y",  # Confirm removal
        MOCK_OPENAI_API_KEY,  # OpenAI API Key
    ]
    # Pre-create the install directory to simulate existing path
    os.makedirs(MOCK_INSTALL_DIR)
    installer = MacosInstaller(MockMacosEnvironment())
    installer.install()
    assert installer.environment.mem_machine_installed
    assert os.path.exists(MOCK_INSTALL_DIR)


@patch("builtins.input")
def test_path_already_exists_cancel_install(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        "n",  # Cancel uninstall
    ]
    # Pre-create the install directory to simulate existing path
    os.makedirs(MOCK_INSTALL_DIR)
    installer = MacosInstaller(MockMacosEnvironment())
    # expect an Exception due to cancellation
    try:
        installer.install()
        assert False, "Expected Exception due to cancellation"
    except Exception as e:
        assert str(e) == "Installation aborted by user."
        assert os.path.exists(MOCK_INSTALL_DIR)


@patch("builtins.input")
def test_path_already_exists_cancel_removal(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        "y",  # Confirm uninstall
        "n",  # Cancel removal
    ]
    # Pre-create the install directory to simulate existing path
    os.makedirs(MOCK_INSTALL_DIR)
    installer = MacosInstaller(MockMacosEnvironment())
    # expect an Exception due to cancellation
    try:
        installer.install()
        assert False, "Expected Exception due to cancellation"
    except Exception as e:
        assert str(e) == "Operation cancelled by user."
        assert os.path.exists(MOCK_INSTALL_DIR)


class MockWindowsEnvironment(WindowsEnvironment):
    def __init__(self):
        super().__init__()
        self.openjdk_zip_downloaded = False
        self.neo4j_zip_downloaded = False
        self.openjdk_extracted = False
        self.neo4j_extracted = False
        self.neo4j_installed = False
        self.neo4j_uninstalled = False

    def download_file(self, url: str, dest: str):
        if url == JDK_URL:
            assert dest == os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME)
            os.open(dest, os.O_CREAT)  # Create an empty file to simulate download
            self.openjdk_zip_downloaded = True
        elif url == NEO4J_URL:
            assert dest == os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME)
            os.open(dest, os.O_CREAT)  # Create an empty file to simulate download
            self.neo4j_zip_downloaded = True
        else:
            raise ValueError("Unexpected URL")

    def extract_zip(self, zip_path: str, extract_to: str):
        assert extract_to == MOCK_INSTALL_DIR
        if zip_path == os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME):
            assert self.openjdk_zip_downloaded
            self.openjdk_extracted = True
        elif zip_path == os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME):
            assert self.neo4j_zip_downloaded
            self.neo4j_extracted = True
        else:
            raise ValueError("Unexpected zip path")

    def install_neo4j_service(self, install_dir: str):
        assert self.neo4j_extracted
        assert self.openjdk_extracted
        self.neo4j_installed = True

    def uninstall_neo4j_service(self, install_dir: str):
        assert install_dir == MOCK_INSTALL_DIR
        self.neo4j_uninstalled = True

    def install_memmachine_in_venv(self, install_dir: str):
        assert install_dir == MOCK_INSTALL_DIR
        assert self.neo4j_installed
        self.mem_machine_installed = True


@patch("builtins.input")
def test_install_in_windows(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        MOCK_OPENAI_API_KEY,  # OpenAI API Key
    ]
    installer = WindowsInstaller(MockWindowsEnvironment())
    installer.install()
    assert installer.environment.mem_machine_installed
    assert os.path.exists(MOCK_INSTALL_DIR)
    assert os.path.exists(os.path.join(MOCK_INSTALL_DIR, "configuration.yml"))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, JDK_ZIP_NAME))
    assert not os.path.exists(os.path.join(MOCK_INSTALL_DIR, NEO4J_ZIP_NAME))
    assert_file_content(
        os.path.join(MOCK_INSTALL_DIR, "configuration.yml"),
        CONFIGURATION_TEMPLATE.format(OPENAI_KEY=MOCK_OPENAI_API_KEY),
    )
    assert os.path.exists(os.path.join(MOCK_INSTALL_DIR, WINDOWS_RUN_SCRIPT_NAME))
    assert_file_content(
        os.path.join(MOCK_INSTALL_DIR, WINDOWS_RUN_SCRIPT_NAME),
        POWERSHELL_STARTUP_TEMPLATE.format(OPENAI_KEY=MOCK_OPENAI_API_KEY),
    )


@patch("builtins.input")
def test_install_in_windows_existing_neo4j_service(mock_input, fs):
    mock_input.side_effect = [
        MOCK_INSTALL_DIR,  # Install directory
        "y",  # Confirm uninstall
        "y",  # Confirm removal
        MOCK_OPENAI_API_KEY,  # OpenAI API Key
    ]
    os.makedirs(MOCK_INSTALL_DIR)  # Simulate existing installation
    installer = WindowsInstaller(MockWindowsEnvironment())
    installer.install()
    assert installer.environment.mem_machine_installed
    assert installer.environment.neo4j_uninstalled

@patch("builtins.input")
def test_install_dir_default_used(mock_input, fs):
    mock_input.side_effect = [
        "",  # Use default install directory
        MOCK_OPENAI_API_KEY,  # OpenAI API Key
    ]
    installer = MacosInstaller(MockMacosEnvironment())
    installer.default_install_dir = MOCK_INSTALL_DIR  # Set default for test
    installer.install()
    assert installer.install_dir == MOCK_INSTALL_DIR
    assert installer.environment.mem_machine_installed
    assert os.path.exists(MOCK_INSTALL_DIR)

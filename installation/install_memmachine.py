import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from typing import Dict

WINDOWS_DEFAULT_INSTALL_DIR = "C:\\Program Files\\MemMachine"
MACOS_DEFAULT_INSTALL_DIR = "/usr/local/memmachine"
JDK_URL = "https://download.oracle.com/java/21/latest/jdk-21_windows-x64_bin.zip"
NEO4J_URL = "https://dist.neo4j.org/neo4j-community-2025.09.0-windows.zip"
JDK_ZIP_NAME = "jdk-21_windows-x64_bin.zip"
NEO4J_ZIP_NAME = "neo4j-community-2025.09.0-windows.zip"
MEMMACHINE_ZIP_NAME = "MemMachine.zip"
WINDOWS_RUN_SCRIPT_NAME = "run_memmachine.ps1"
MACOS_RUN_SCRIPT_NAME = "run_memmachine.sh"

JDK_DIR_NAME = "jdk-21.0.9"
NEO4J_DIR_NAME = "neo4j-community-2025.09.0"

CONFIGURATION_TEMPLATE = """\
logging:
  path: /tmp/memory_log
  level: debug #| debug | error

long_term_memory:
  derivative_deriver: sentence
  metadata_prefix: "[$timestamp] $producer_id: "
  embedder: my_embedder_id
  reranker: my_reranker_id
  vector_graph_store: my_storage_id

SessionDB:
  uri: sqlitetest.db

Model:
  testmodel:
    model_vendor: openai
    model_name: "gpt-4.1-mini"
    api_key: {OPENAI_KEY}

storage:
  my_storage_id:
    vendor_name: neo4j
    host: localhost
    port: 7687
    user: neo4j
    password: memmachine

  profile_storage:
    vendor_name: postgres
    host: localhost
    port: 5432
    user: memmachine
    db_name: memmachine
    password: memmachine_password

profile_memory:
  llm_model: testmodel
  embedding_model: my_embedder_id
  database: profile_storage
  prompt: profile_prompt

sessionMemory:
  model_name: testmodel
  message_capacity: 500
  max_message_length: 16000
  max_token_num: 8000

embedder:
  my_embedder_id:
    name: openai
    config:
      model_name: "gpt-4.1-mini"
      api_key: {OPENAI_KEY}

reranker:
  my_reranker_id:
    type: "rrf-hybrid"
    reranker_ids:
      - id_ranker_id
      - bm_ranker_id
  id_ranker_id:
    type: "identity"
  bm_ranker_id:
    type: "bm25"

prompt:
  profile: profile_prompt
"""

BASH_STARTUP_TEMPLATE = """\
# Database configuration
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USER="memmachine"
export POSTGRES_PASSWORD="memmachine_password"
export POSTGRES_DB="memmachine"

# Neo4j configuration
export NEO4J_HOST="neo4j"
export NEO4J_PORT="7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4j"

# Application configuration
export MEMORY_CONFIG="configuration.yml"
export MCP_BASE_URL="http://localhost:8080"
export GATEWAY_URL="http://localhost:8080"
export FAST_MCP_LOG_LEVEL="INFO"
export OPENAI_API_KEY="{OPENAI_KEY}"
export LOG_LEVEL="INFO"

source .venv/bin/activate
memmachine-sync-profile-schema
memmachine-server
"""

POWERSHELL_STARTUP_TEMPLATE = """\
# Database configuration
$env:POSTGRES_HOST="localhost"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_USER="memmachine"
$env:POSTGRES_PASSWORD="memmachine_password"
$env:POSTGRES_DB="memmachine"

# Neo4j configuration
$env:NEO4J_HOST="neo4j"
$env:NEO4J_PORT="7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="neo4j"

# Application configuration
$env:MEMORY_CONFIG="configuration.yml"
$env:MCP_BASE_URL="http://localhost:8080"
$env:GATEWAY_URL="http://localhost:8080"
$env:FAST_MCP_LOG_LEVEL="INFO"
$env:OPENAI_API_KEY="{OPENAI_KEY}"
$env:LOG_LEVEL="INFO"

.\\.venv\\Scripts\\Activate.ps1
memmachine-sync-profile-schema.exe
memmachine-server.exe
"""


class Installer(ABC):
    def __init__(self):
        self.install_dir = ""
        self.default_install_dir = ""
        self.run_script_name = ""
        self.run_script_template = ""

    @abstractmethod
    def install_memmachine_in_venv(self):
        pass

    @abstractmethod
    def install_neo4j(self):
        pass

    @abstractmethod
    def uninstall(self):
        pass

    def generate_run_script(self, api_key: str):
        with open(
            os.path.join(self.install_dir, self.run_script_name), "w", encoding="utf-8"
        ) as run_file:
            run_content = self.run_script_template.format(OPENAI_KEY=api_key)
            run_file.write(run_content)

    def generate_configuration(self, api_key: str):
        with open(
            os.path.join(self.install_dir, "configuration.yml"), "w", encoding="utf-8"
        ) as config_file:
            config_content = CONFIGURATION_TEMPLATE.format(OPENAI_KEY=api_key)
            config_file.write(config_content)

    def remove_directory(self, dir_path: str):
        print(f"Removing directory {dir_path}, Proceed? (y/n): ", end="")
        choice = input().lower()
        if choice != "y":
            raise Exception("Operation cancelled by user.")
        shutil.rmtree(dir_path)

    def ask_install_dir(self):
        user_input = input(f"Please enter installation directory (default: {self.default_install_dir}): ").strip()
        self.install_dir = user_input or self.default_install_dir

    def install(self):
        self.ask_install_dir()
        # make sure the directory does not already exist
        if os.path.exists(self.install_dir):
            print(
                f"Installation directory {self.install_dir} already exists. Would you like to uninstall first? (y/n): ",
                end="",
            )
            choice = input().lower()
            if choice != "y":
                raise Exception("Installation aborted by user.")
            self.uninstall()
        os.makedirs(self.install_dir, exist_ok=True)
        print(f"Installing to {self.install_dir}...")

        print("Installing Neo4j...")
        self.install_neo4j()

        print("Installing MemMachine...")
        self.install_memmachine_in_venv()

        # ask for OpenAI API key
        print("Please enter your OpenAI API key: ", end="")
        openai_key = input().strip()

        print("Generating configuration file and run script...")
        self.generate_configuration(openai_key)
        self.generate_run_script(openai_key)
        print(
            f"you can now run MemMachine by executing the {self.run_script_name} script at {os.path.join(self.install_dir, self.run_script_name)}."
        )

        print("Installation completed successfully.")


class MacosEnvironment:
    def install_memmachine_in_venv(self, install_dir: str):
        subprocess.run(["python3", "-m", "venv", f"{install_dir}/.venv"], check=True)
        subprocess.run(
            [
                "bash",
                "-c",
                f"source {install_dir}/.venv/bin/activate && pip install memmachine",
            ],
            check=True,
        )

    def install_neo4j(self):
        subprocess.run(["brew", "install", "neo4j"], check=True)
        subprocess.run(["brew", "services", "start", "neo4j"], check=True)

    def neo4j_installed(self) -> bool:
        result = subprocess.run(
            ["brew", "list", "--versions", "neo4j"], capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() != ""


class MacosInstaller(Installer):
    def __init__(self, environment: MacosEnvironment = MacosEnvironment()):
        super().__init__()
        self.default_install_dir = MACOS_DEFAULT_INSTALL_DIR
        self.run_script_name = MACOS_RUN_SCRIPT_NAME
        self.run_script_template = BASH_STARTUP_TEMPLATE
        self.environment = environment

    def install_memmachine_in_venv(self):
        self.environment.install_memmachine_in_venv(self.install_dir)

    def install_neo4j(self):
        if self.environment.neo4j_installed():
            print("Neo4j is already installed. Skipping installation.")
            return
        self.environment.install_neo4j()

    def uninstall(self):
        print("Uninstalling...")
        self.remove_directory(self.install_dir)


class WindowsEnvironment:
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

    def install_neo4j_service(self, install_dir: str):
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

    def get_neo4j_env(self, install_dir: str) -> Dict[str, str]:
        return {
            "JAVA_HOME": os.path.join(install_dir, JDK_DIR_NAME),
            "NEO4J_HOME": os.path.join(install_dir, NEO4J_DIR_NAME),
        }

    def uninstall_neo4j_service(self, install_dir: str):
        neo4j_bin = os.path.join(install_dir, NEO4J_DIR_NAME, "bin", "neo4j.ps1")
        if not os.path.exists(neo4j_bin):
            print("Neo4j installation not found; skipping service uninstallation.")
            return
        subprocess.run(
            ["powershell.exe", "-File", neo4j_bin, "windows-service", "uninstall"],
            env={**os.environ.copy(), **self.get_neo4j_env(install_dir)},
            check=True,
        )

    def install_memmachine_in_venv(self, install_dir: str):
        subprocess.run(["python", "-m", "venv", f"{install_dir}\\.venv"], check=True)
        subprocess.run(
            [
                "powershell.exe",
                "-Command",
                f"{install_dir}\\.venv\\Scripts\\Activate.ps1; pip install memmachine",
            ],
            check=True,
        )


class WindowsInstaller(Installer):
    def __init__(self, environment: WindowsEnvironment = WindowsEnvironment()):
        super().__init__()
        self.default_install_dir = WINDOWS_DEFAULT_INSTALL_DIR
        self.run_script_name = WINDOWS_RUN_SCRIPT_NAME
        self.run_script_template = POWERSHELL_STARTUP_TEMPLATE
        self.environment = environment

    def install_memmachine_in_venv(self):
        self.environment.install_memmachine_in_venv(self.install_dir)

    def install_neo4j(self):
        print("Downloading and installing OpenJDK 21...")
        jdk_zip_path = os.path.join(self.install_dir, JDK_ZIP_NAME)
        self.environment.download_file(JDK_URL, jdk_zip_path)
        self.environment.extract_zip(jdk_zip_path, self.install_dir)
        print("OpenJDK 21 installed successfully.")
        print("Downloading and installing Neo4j Community Edition 2025.09.0...")
        neo4j_zip_path = os.path.join(self.install_dir, NEO4J_ZIP_NAME)
        self.environment.download_file(NEO4J_URL, neo4j_zip_path)
        self.environment.extract_zip(neo4j_zip_path, self.install_dir)
        print("Neo4j Community Edition installed successfully.")
        # delete zip files
        os.remove(jdk_zip_path)
        os.remove(neo4j_zip_path)
        # install and start neo4j service
        print("Starting Neo4j service...")
        self.environment.install_neo4j_service(self.install_dir)
        print("Neo4j service started.")

    def uninstall(self):
        print("Uninstalling...")
        self.environment.uninstall_neo4j_service(self.install_dir)
        self.remove_directory(self.install_dir)


def main():
    try:
        print("Installing...")
        system = platform.system()
        if system == "Windows":
            WindowsInstaller().install()
        elif system == "Darwin":
            MacosInstaller().install()
        else:
            raise Exception(f"Unsupported operating system: {system}")
    except Exception as e:
        print(f"Installation failed: {e}")


if __name__ == "__main__":
    main()

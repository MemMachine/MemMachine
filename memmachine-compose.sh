#!/bin/bash

# Ensure the script exits immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# The return value of a pipeline is the exit status of the last command in the pipeline that failed,
# or zero if no command failed.
set -euo pipefail

# Configuration files
CONFIG_FILE="configuration.yml"
ENV_FILE=".env"
SAMPLE_CONFIG_FILE="sample_configs/episodic_memory_config.sample"

# --- Colors for output ---
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

# --- Logging Functions ---
log_info() {
    echo -e "${COLOR_BLUE}INFO:${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_YELLOW}WARN:${COLOR_RESET} $1"
}

log_error() {
    echo -e "${COLOR_RED}ERROR:${COLOR_RESET} $1" >&2
}

log_success() {
    echo -e "${COLOR_GREEN}SUCCESS:${COLOR_RESET} $1"
}

# --- Utility Functions ---

# Check for yq (required for robust YAML parsing)
check_yq_installed() {
    log_info "Checking for 'yq' tool..."
    if ! command -v yq &> /dev/null; then
        log_error "The 'yq' tool is required for robust YAML configuration management."
        log_error "Please install it. For example, using Homebrew (macOS/Linux): 'brew install yq'"
        log_error "Or via pip: 'pip install yq' (Python version)."
        log_error "Alternatively, download from GitHub releases: https://github.com/mikefarah/yq/releases"
        exit 1
    fi
    log_success "'yq' found."
}

# Function to get a scalar value from configuration.yml using yq
# Expects a path like ".neo4j_db_instance.host"
get_config_value() {
    local key_path="$1"
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file '$CONFIG_FILE' not found. Cannot retrieve value for '$key_path'."
        exit 1
    fi
    # Use yq to read the value. Trim whitespace, ensure no newline.
    # The 'e' flag makes yq exit with 1 if the path doesn't exist.
    # We suppress stderr here as yq's error message isn't always user-friendly for "path not found".
    yq e "$key_path" "$CONFIG_FILE" 2>/dev/null | tr -d '\r' || {
        # If yq e returns non-zero, it means the key_path was not found or there was an error.
        log_warn "Could not retrieve '$key_path' from '$CONFIG_FILE' (it might be missing or malformed)."
        echo "" # Return an empty string
    }
}

# Function to update a scalar value in configuration.yml using yq
# Expects a path like ".neo4j_db_instance.host"
update_config_value() {
    local key_path="$1"
    local new_val="$2"
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file '$CONFIG_FILE' not found. Cannot update '$key_path'."
        exit 1
    fi
    # Use yq to write the value in-place
    yq e -i "$key_path = \"$new_val\"" "$CONFIG_FILE" || {
        log_error "Failed to update '$key_path' in '$CONFIG_FILE' using yq."
        return 1
    }
    return 0
}

# Function to get a scalar value from .env
get_env_value() {
    local key="$1"
    if [ -f "$ENV_FILE" ]; then
        # Use awk to handle potential comments on the same line and trim whitespace
        # Ensure only the first match is returned
        awk -F= -v key_name="$key" '$1 == key_name {gsub(/^ *| *$/, "", $2); print $2; exit}' "$ENV_FILE" | tr -d '\r'
    fi
}

# --- Core Script Functions ---

# Check if Docker and Docker Compose are available
check_docker_compose() {
    log_info "Checking Docker and Docker Compose installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    local docker_compose_cmd_v1="docker-compose"
    local docker_compose_cmd_v2="docker compose"

    if command -v "${docker_compose_cmd_v1}" &> /dev/null; then
        DOCKER_COMPOSE_CMD="${docker_compose_cmd_v1}"
    elif command -v "${docker_compose_cmd_v2}" &> /dev/null; then
        DOCKER_COMPOSE_CMD="${docker_compose_cmd_v2}"
    else
        log_error "Docker Compose (V1 or V2) is not installed. Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    log_success "Docker and Docker Compose ('${DOCKER_COMPOSE_CMD}') are available."
}

# Check if configuration.yml exists, if not, create it from sample
check_config_file_exists() {
    log_info "Checking for $CONFIG_FILE..."
    if [ ! -f "$CONFIG_FILE" ]; then
        log_warn "$CONFIG_FILE not found. Creating from $SAMPLE_CONFIG_FILE."
        if [ ! -f "$SAMPLE_CONFIG_FILE" ]; then
            log_error "$SAMPLE_CONFIG_FILE not found. Cannot create $CONFIG_FILE. Exiting."
            exit 1
        fi
        cp "$SAMPLE_CONFIG_FILE" "$CONFIG_FILE"
        log_success "$CONFIG_FILE created successfully from sample."
    else
        log_success "$CONFIG_FILE found."
    fi
}

# Check if .env exists and contains OPENAI_API_KEY
check_openai_api_key() {
    log_info "Checking for $ENV_FILE and OpenAI API key..."
    if [ ! -f "$ENV_FILE" ]; then
        log_error "$ENV_FILE not found. Please copy 'sample_configs/env.dockercompose' to '.env' and configure your OpenAI API key."
        exit 1
    fi

    local openai_api_key_env
    openai_api_key_env=$(get_env_value "OPENAI_API_KEY")
    if [ -z "$openai_api_key_env" ]; then
        log_error "OPENAI_API_KEY is not set in $ENV_FILE. Please add your OpenAI API key."
        exit 1
    fi
    log_success "OpenAI API key found in $ENV_FILE."
}

# Function to set up and validate configuration for Docker
setup_config_and_env_for_docker() {
    log_info "Starting Docker environment configuration and validation..."

    # 1. Create a backup of configuration.yml
    local backup_file="${CONFIG_FILE}.bak.$(date +%Y%m%d%H%M%S)"
    cp "$CONFIG_FILE" "$backup_file"
    log_info "Backup of $CONFIG_FILE created at $backup_file"

    # 2. Extract NEO4J_PASSWORD from .env
    local neo4j_password_env
    neo4j_password_env=$(get_env_value "NEO4J_PASSWORD")
    if [ -z "$neo4j_password_env" ]; then
        log_error "NEO4J_PASSWORD is not set in $ENV_FILE."
        log_error "Please add your Neo4j password to .env. Cannot proceed without it."
        exit 1
    fi
    log_success "NEO4J_PASSWORD found in $ENV_FILE."

    # 3. Read Neo4j host and password from configuration.yml
    local neo4j_host_config
    neo4j_host_config=$(get_config_value ".neo4j_db_instance.host")
    if [ -z "$neo4j_host_config" ]; then
        log_error "Neo4j host not found or is empty in $CONFIG_FILE under '.neo4j_db_instance.host'."
        log_error "Please ensure 'host:' is properly defined in the 'neo4j_db_instance' section of $CONFIG_FILE."
        exit 1
    fi

    local neo4j_password_config
    neo4j_password_config=$(get_config_value ".neo4j_db_instance.password")
    if [ -z "$neo4j_password_config" ]; then
        log_error "Neo4j password not found or is empty in $CONFIG_FILE under '.neo4j_db_instance.password'."
        log_error "Please ensure 'password:' is properly defined in the 'neo4j_db_instance' section of $CONFIG_FILE."
        exit 1
    fi

    # 4. Auto-correction: host: localhost to host: neo4j
    if [ "${neo4j_host_config}" == "localhost" ]; then
        log_warn "Neo4j host in $CONFIG_FILE is 'localhost'. Automatically correcting to 'neo4j' for Docker Compose."
        if update_config_value ".neo4j_db_instance.host" "neo4j"; then
            log_success "Neo4j host updated to 'neo4j' in $CONFIG_FILE."
            neo4j_host_config="neo4j" # Update variable after change
        else
            log_error "Failed to auto-correct Neo4j host in $CONFIG_FILE. Manual intervention required or check yq errors."
            exit 1 # Fatal error if auto-correction fails
        fi
    fi

    # 5. Hostname Validation
    if [ "${neo4j_host_config}" != "neo4j" ]; then
        log_error "Neo4j host in $CONFIG_FILE is set to '${neo4j_host_config}'. For Docker Compose, it *must* be 'neo4j'."
        log_error "Please edit $CONFIG_FILE and set 'host: neo4j' under 'neo4j_db_instance:'."
        exit 1
    else
        log_success "Neo4j host in $CONFIG_FILE is correctly set to 'neo4j'."
    fi

    # 6. Password Validation
    if [ "${neo4j_password_config}" == "<YOUR_PASSWORD_HERE>" ]; then
         log_error "Neo4j password in $CONFIG_FILE is still '<YOUR_PASSWORD_HERE>'."
         log_error "Please update it to match the NEO4J_PASSWORD in .env."
         exit 1
    fi
    if [ "${neo4j_password_config}" != "${neo4j_password_env}" ]; then
        log_error "Neo4j password in $CONFIG_FILE does NOT match NEO4J_PASSWORD in $ENV_FILE."
        log_error "Please ensure both are consistent. Check the values in both files."
        exit 1
    else
        log_success "Neo4j password in $CONFIG_FILE matches NEO4J_PASSWORD in $ENV_FILE."
    fi

    log_success "Docker environment configuration and validation complete."
}

# --- Service Control Functions ---

# Start Docker Compose services
start_services() {
    log_info "Preparing to start MemMachine services..."
    check_docker_compose
    check_yq_installed
    check_config_file_exists
    check_openai_api_key
    setup_config_and_env_for_docker # New validation and auto-correction
    
    log_info "Starting MemMachine Docker Compose services..."
    ${DOCKER_COMPOSE_CMD} up -d
    if [ $? -ne 0 ]; then
        log_error "Failed to start Docker Compose services."
        exit 1
    fi

    log_info "Waiting for services to become healthy (approximately 10 seconds)..."
    sleep 10 # Give services some time to start. A proper health check loop would be more robust.

    log_success "MemMachine services are running."
    log_info "You can access:"
    log_info "  MemMachine API: http://localhost:8080"
    log_info "  Neo4j Browser: http://localhost:7474 (Use username 'neo4j' and your configured password)"
    log_info "  Health Check: http://localhost:8080/health"
    log_info "  Metrics: http://localhost:8080/metrics"
}

# Stop Docker Compose services
stop_services() {
    log_info "Stopping MemMachine services..."
    check_docker_compose # Only need Docker Compose check to stop services
    ${DOCKER_COMPOSE_CMD} down
    if [ $? -ne 0 ]; then
        log_warn "Failed to stop Docker Compose services gracefully. They might not have been running or an issue occurred."
    fi
    log_success "MemMachine services stopped."
}

# Restart Docker Compose services
restart_services() {
    log_info "Preparing to restart MemMachine services..."
    check_docker_compose
    check_yq_installed
    check_config_file_exists
    check_openai_api_key
    setup_config_and_env_for_docker # Re-validate on restart
    
    log_info "Restarting MemMachine Docker Compose services..."
    ${DOCKER_COMPOSE_CMD} restart
    if [ $? -ne 0 ]; then
        log_error "Failed to restart Docker Compose services."
        exit 1
    fi
    log_info "Waiting for services to become healthy (approximately 10 seconds)..."
    sleep 10 # Give services some time to start
    log_success "MemMachine services restarted."
}

# Clean up (remove containers, networks, and volumes)
clean_up() {
    log_info "Initiating cleanup of MemMachine Docker environment (this will remove all data)..."
    read -rp "${COLOR_YELLOW}Are you sure you want to remove all MemMachine data and containers? (y/N): ${COLOR_RESET}" -n 1 -r
    echo # Newline after read -n 1
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled."
        return 0
    fi

    check_docker_compose # Only need Docker Compose check for cleanup
    log_info "Removing Docker containers, networks, and volumes..."
    ${DOCKER_COMPOSE_CMD} down -v
    if [ $? -ne 0 ]; then
        log_error "Failed to clean up Docker Compose environment."
        exit 1
    fi
    log_success "MemMachine Docker environment cleaned up. All data removed."
}

# Show Docker Compose logs
show_logs() {
    log_info "Showing MemMachine service logs (Ctrl+C to exit)..."
    check_docker_compose # Only need Docker Compose check to show logs
    ${DOCKER_COMPOSE_CMD} logs -f
}

# Show help message
show_help() {
    echo "Usage: ./memmachine-compose.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start MemMachine services (PostgreSQL, Neo4j, MemMachine API)."
    echo "            Performs configuration validation and auto-correction."
    echo "  stop      Stop MemMachine services."
    echo "  restart   Restart MemMachine services."
    echo "  logs      View real-time logs for all services."
    echo "  clean     Remove all MemMachine Docker containers, networks, and volumes (deletes all data!)."
    echo "  help      Show this help message."
    echo ""
    echo "Configuration files:"
    echo "  $ENV_FILE: Contains environment variables (e.g., OPENAI_API_KEY, NEO4J_PASSWORD)."
    echo "  $CONFIG_FILE: MemMachine application configuration (Neo4j host, password)."
    echo ""
    echo "Important: When using 'start' or 'restart', the script will:"
    echo "  - Create a backup of '$CONFIG_FILE'."
    echo "  - Automatically correct Neo4j 'host: localhost' to 'host: neo4j' in '$CONFIG_FILE'."
    echo "  - Validate Neo4j host and password consistency."
    echo "  - Requires 'yq' for robust YAML parsing. See 'yq' installation instructions if prompted."
}

# Handle command-line arguments
handle_command() {
    if [ $# -eq 0 ]; then
        log_error "No command provided."
        show_help
        exit 1
    fi

    case "$1" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_up
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Invalid command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Main execution
handle_command "$@"
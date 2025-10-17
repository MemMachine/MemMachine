#!/usr/bin/env bash

# MemMachine Docker Startup Script
# This script helps you get MemMachine running with Docker Compose

set -e

set -x

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

## Function to run a command with a timeout
timeout() {
    local duration=$1
    shift

    # Run the command in the background
    "$@" &
    local cmd_pid=$!

    # Start a background sleep that will kill the command
    (
        sleep "$duration"
        kill -0 "$cmd_pid" 2>/dev/null && kill -TERM "$cmd_pid"
    ) &

    local watchdog_pid=$!

    # Wait for the command to finish
    wait "$cmd_pid"
    local status=$?

    # Clean up watchdog if command finished early
    kill -TERM "$watchdog_pid" 2>/dev/null

    return $status
}

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_prompt() {
    echo -ne "${MAGENTA}[PROMPT]${NC} "
}

safe_sed_inplace() {
    if sed --version >/dev/null 2>&1; then
        # GNU/Linux sed
        sed -i "$1" "$2"
    else
        # BSD/macOS sed
        sed -i '' "$1" "$2"
    fi
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        sleep 1
        if [ -f "sample_configs/env.dockercompose" ]; then
            cp sample_configs/env.dockercompose .env
            print_success "Created .env file from sample_configs/env.dockercompose"
        else
            print_error "sample_configs/env.dockercompose file not found. Please create .env file manually."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# In lieu of yq, use awk to read over the configuration.yml file line-by-line,
# and set the database credentials using the same environment variables as in docker-compose.yml
set_config_defaults() {
    awk -v pg_user="${POSTGRES_USER:-memmachine}" \
        -v pg_db="${POSTGRES_DB:-memmachine}" \
        -v pg_pass="${POSTGRES_PASS:-memmachine_password}" \
        -v neo4j_user="${NEO4J_USER:-neo4j}" \
        -v neo4j_pass="${NEO4J_PASSWORD:-neo4j_password}" '
/vendor_name:/ {
  vendor = $2
}

vendor == "neo4j" && /host:/ { sub(/localhost/, "neo4j") }
vendor == "neo4j" && /password:/ { sub(/<YOUR_PASSWORD_HERE>/, neo4j_pass) }

vendor == "postgres" && /host:/ { sub(/localhost/, "postgres") }
vendor == "postgres" && /user:/ { sub(/postgres/, pg_user) }
vendor == "postgres" && /db_name:/ { sub(/postgres/, pg_db) }
vendor == "postgres" && /password:/ { sub(/<YOUR_PASSWORD_HERE>/, pg_pass) }

{ print }
' configuration.yml > configuration.yml.tmp && mv configuration.yml.tmp configuration.yml
}

# Check if configuration.yml file exists
check_config_file() {
    if [ ! -f "configuration.yml" ]; then
        print_warning "configuration.yml file not found. Creating from template..."
        sleep 1

        # Ask user for CPU or GPU configuration, defaulting to CPU
        print_prompt
        read -p "Which configuration would you like to use for the Docker Image? (CPU/GPU) [CPU]: " config_type_input
        local config_type=$(echo "${config_type_input:-CPU}" | tr '[:lower:]' '[:upper:]')

        if [ "$config_type" = "GPU" ]; then
            CONFIG_SOURCE="sample_configs/episodic_memory_config.gpu.sample"
            MEMMACHINE_IMAGE="memmachine/memmachine:latest-gpu"
            print_info "GPU configuration selected."
        else
            if [ -n "$config_type_input" ] && [ "$config_type" != "CPU" ]; then
                print_warning "Invalid selection. Defaulting to CPU."
            else
                print_info "CPU configuration selected."
            fi
            CONFIG_SOURCE="sample_configs/episodic_memory_config.cpu.sample"
            MEMMACHINE_IMAGE="memmachine/memmachine:latest-cpu"
        fi

        # Update .env file with the selected image
        if [ -f ".env" ]; then
            # Remove existing MEMMACHINE_IMAGE from .env if it exists
            safe_sed_inplace '/^MEMMACHINE_IMAGE=/d' .env
        fi
        echo "MEMMACHINE_IMAGE=${MEMMACHINE_IMAGE}" >> .env
        print_success "Set MEMMACHINE_IMAGE to ${MEMMACHINE_IMAGE} in .env file"

        if [ -f "$CONFIG_SOURCE" ]; then
            cp "$CONFIG_SOURCE" configuration.yml
            print_success "Created configuration.yml file from $CONFIG_SOURCE"
        else
            print_error "$CONFIG_SOURCE file not found. Please create configuration.yml file manually."
            exit 1
        fi

        set_config_defaults
    else
        print_success "configuration.yml file found"
    fi
}

# Prompt user if they would like to set their OpenAI API key; then set it in the .env file and configuration.yml file
set_openai_api_key() {
    local api_key=""
    local reply=""
    if [ -f ".env" ]; then
        source .env
        if [ -z "$OPENAI_API_KEY" ] ||  [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ] || grep -q "<YOUR_API_KEY>" configuration.yml ; then
            print_prompt
            read -p "OPENAI_API_KEY is not set or is using placeholder value. Would you like to set your OpenAI API key? (y/N) " reply
            if [[ $reply =~ ^[Yy]$ ]]; then
                print_prompt
                read -sp "Enter your OpenAI API key: " api_key
                echo setting .env
                safe_sed_inplace "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
                echo setting configuration.yml
                safe_sed_inplace "s/api_key: .*$/api_key: $api_key/g" configuration.yml
                print_success "Set OPENAI_API_KEY in .env and configuration.yml"
            fi
        fi
    fi
}

# Check if required environment variables are set
check_required_env() {
    if [ -f ".env" ]; then
        source .env
        
        if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
            print_warning "OPENAI_API_KEY is not set or is using placeholder value"
            print_warning "Please set your OpenAI API key in the .env file"
            print_prompt
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "OPENAI_API_KEY is configured"
        fi
    fi
}

# Check if configuration.yml has required fields
check_required_config() {
    if [ -f "configuration.yml" ]; then
        # Check for API key in configuration.yml - look for actual placeholder patterns
        if grep -q "api_key.*your_.*_api_key_here" configuration.yml || grep -q "api_key.*sk-example" configuration.yml || grep -q "api_key.*sk-test" configuration.yml; then
            print_warning "API key in configuration.yml appears to be a placeholder or example value"
            print_warning "Please set your actual API key in the configuration.yml file"
            print_prompt
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "API key in configuration.yml appears to be configured"
        fi
        
        # Check for database credentials - look for generic placeholder passwords
        if grep -q "password.*password" configuration.yml && ! grep -q "password.*memmachine_password" configuration.yml; then
            print_warning "Database password in configuration.yml appears to be a placeholder"
            print_warning "Please set your actual database password in the configuration.yml file"
            print_prompt
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "Database credentials in configuration.yml appear to be configured"
        fi
    fi
}

# Pull and start services
start_services() {
    local memmachine_image_tmp="${ENV_MEMMACHINE_IMAGE:-}"

    print_info "Pulling and starting MemMachine services..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    # Unset the memmachine image temporarily; without this, 'docker compose pull' will attempt
    # to pull ${MEMMACHINE_IMAGE} if it is set, which may not be a remote image.
    ENV_MEMMACHINE_IMAGE=""
    # Pull the latest images to ensure we are running the latest version
    print_info "Pulling latest images..."
    $COMPOSE_CMD pull
    ENV_MEMMACHINE_IMAGE="${memmachine_image_tmp:-}"

    # Start services (override the image if specified in memmachine-compose.sh start <image>:<tag>)
    print_info "Starting containers..."
    MEMMACHINE_IMAGE="${ENV_MEMMACHINE_IMAGE:-}" $COMPOSE_CMD up -d
    
    print_success "Services started successfully!"
}

# Wait for services to be healthy
wait_for_health() {
    print_info "Waiting for services to be healthy..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Wait for services to be healthy
    $COMPOSE_CMD ps
    
    print_info "Checking service health..."
    
    # Wait for PostgreSQL
    print_info "Waiting for PostgreSQL to be ready..."
    if timeout 120 bash -c "until docker exec memmachine-postgres pg_isready -U ${POSTGRES_USER:-memmachine} -d ${POSTGRES_DB:-memmachine}; do sleep 2; done"; then
        print_success "PostgreSQL is ready"
    else
        print_error "PostgreSQL failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
    
    # Wait for Neo4j
    print_info "Waiting for Neo4j to be ready..."
    if timeout 120 bash -c "until docker exec memmachine-neo4j cypher-shell -u ${NEO4J_USER:-neo4j} -p ${NEO4J_PASSWORD:-neo4j_password} 'RETURN 1' > /dev/null 2>&1; do sleep 2; done"; then
        print_success "Neo4j is ready"
    else
        print_error "Neo4j failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
    
    # Wait for MemMachine
    print_info "Waiting for MemMachine to be ready..."
    if timeout 120 bash -c "until curl -f http://localhost:${MEMORY_SERVER_PORT:-8080}/health > /dev/null 2>&1; do sleep 5; done"; then
        print_success "MemMachine is ready"
    else
        print_error "MemMachine failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
}

# Show service information
show_service_info() {
    print_success "🎉 MemMachine is now running!"
    echo ""
    echo "Service URLs:"
    echo "  📊 MemMachine API: http://localhost:${MEMORY_SERVER_PORT:-8080}"
    echo "  🗄️  Neo4j Browser: http://localhost:${NEO4J_HTTP_PORT:-7474}"
    echo "  📈 Health Check: http://localhost:${MEMORY_SERVER_PORT:-8080}/health"
    echo "  📊 Metrics: http://localhost:${MEMORY_SERVER_PORT:-8080}/metrics"
    echo ""
    echo "Database Access:"
    echo "  🐘 PostgreSQL: localhost:${POSTGRES_PORT:-5432} (user: ${POSTGRES_USER:-memmachine}, db: ${POSTGRES_DB:-memmachine})"
    echo "  🔗 Neo4j Bolt: localhost:${NEO4J_PORT:-7687} (user: ${NEO4J_USER:-neo4j})"
    echo ""
    echo "Useful Commands:"
    echo "  📋 View logs: docker-compose logs -f"
    echo "  🛑 Stop services: docker-compose down"
    echo "  🔄 Restart: docker-compose restart"
    echo "  🧹 Clean up: docker-compose down -v"
    echo ""
}

build_image() {
    local name=""
    local force="false"
    local gpu="false"
    local reply=""
    local key=""
    local value=""

    while [[ $# -gt 0 ]]; do
        # This section splits the key and value if they are separated by an "=" sign
        if [[ "$1" == --* ]]; then
            if [[ "$1" == *=* ]]; then
                key=$(echo "$1" | cut -d '=' -f 1)
                value=$(echo "$1" | cut -d '=' -f 2-)
                shift
            else
                key="$1"
                value="$2"
                if [[ "$#" -ge 2 ]]; then
                    shift 2
                else
                    print_error "Missing value for argument: $1"
                    exit 1
                fi
            fi
        else 
            # If no leading "--", then this is not an option, so just use put the argument in $key
            key="$1"
            value=""
            shift
        fi

        case "$key" in
            --gpu)
                gpu="$value"
                ;;
            -f|--force)
                force="true"
                ;;
            *)
                name="$key"
                ;;
        esac
    done

    if [[ -z "$name" ]]; then
        print_info "No name specified."
        print_info "Using default name: memmachine/memmachine:latest"
        name="memmachine/memmachine:latest"
    fi

    if [[ "$force" == "false" ]]; then
        print_prompt
        read -p "Building $name with '--build-arg GPU=$gpu' (y/N): " -r reply
    else
        print_info "Building $name with '--build-arg GPU=$gpu'"
    fi

    if [[ $reply =~ ^[Yy]$ || $force == "true" ]]; then
        docker build --build-arg GPU=$gpu -t "$name" .
    else
        print_info "Build cancelled"
        exit 0
    fi
}

# Main execution
main() {
    echo "MemMachine Docker Startup Script"
    echo "===================================="
    echo ""
    
    check_docker
    check_env_file
    check_config_file
    set_openai_api_key
    check_required_env
    check_required_config
    start_services
    wait_for_health
    show_service_info
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_info "Stopping MemMachine services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down
        else
            docker compose down
        fi
        print_success "Services stopped"
        ;;
    "restart")
        print_info "Restarting MemMachine services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose restart
        else
            docker compose restart
        fi
        print_success "Services restarted"
        ;;
    "logs")
        print_info "Showing MemMachine logs..."
        if command -v docker-compose &> /dev/null; then
            docker-compose logs -f
        else
            docker compose logs -f
        fi
        ;;
    "clean")
        print_warning "This will remove all data and volumes!"
        print_prompt
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Cleaning up MemMachine services and data..."
            if command -v docker-compose &> /dev/null; then
                docker-compose down -v
            else
                docker compose down -v
            fi
            print_success "Cleanup completed"
        else
            print_info "Cleanup cancelled"
        fi
        ;;
    "build")
        shift
        build_image "$@"
        ;;
    "help"|"-h"|"--help")
        echo "MemMachine Docker Startup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args) | start [<image>:<tag>]                      Start MemMachine services"
        echo "  stop                                                   Stop MemMachine services"
        echo "  restart                                                Restart MemMachine services"
        echo "  logs                                                   Show service logs"
        echo "  clean                                                  Remove all services and data"
        echo "  build [<image>:<tag>] [--gpu true/false] [-f|--force]  Build a custom MemMachine image"
        echo "  help                                                   Show this help message"
        echo ""
        ;;
    "")
        main
        ;;
    "start")
        shift
        ENV_MEMMACHINE_IMAGE="${1:-}"
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

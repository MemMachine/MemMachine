#!/usr/bin/env bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
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

run_with_timeout() {
    local timeout_secs="$1"
    shift
    local start_time
    start_time=$(date +%s)
    while true; do
        if bash -c "$*"; then
            return 0
        fi
        local now
        now=$(date +%s)
        if [ $(( now - start_time )) -ge "$timeout_secs" ]; then
            return 1
        fi
        sleep 2
    done
}

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

check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f "sample_configs/env.dockercompose" ]; then
            cp sample_configs/env.dockercompose .env
            print_success "Created .env file from sample_configs/env.dockercompose"
            print_warning "Please edit .env file with your configuration before continuing"
            print_warning "Especially set your OPENAI_API_KEY"
            print_info "Exiting script. Please edit .env file and re-run the script."
            exit 0
        else
            print_error "sample_configs/env.dockercompose file not found. Please create .env file manually."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

check_config_file() {
    if [ ! -f "configuration.yml" ]; then
        print_warning "configuration.yml file not found. Creating from template..."
        if [ -f "sample_configs/episodic_memory_config.sample" ]; then
            cp sample_configs/episodic_memory_config.sample configuration.yml
            print_success "Created configuration.yml file from sample_configs/episodic_memory_config.sample"
            print_warning "Please edit configuration.yml file with your configuration before continuing"
            print_warning "Especially set your API keys and database credentials"
            print_info "Exiting script. Please edit configuration.yml file and re-run the script."
            exit 0
        else
            print_error "sample_configs/episodic_memory_config.sample file not found. Please create configuration.yml file manually."
            exit 1
        fi
    else
        print_success "configuration.yml file found"
    fi
}

check_required_env() {
    if [ -f ".env" ]; then
        source .env
        
        if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
            print_warning "OPENAI_API_KEY is not set or is using placeholder value"
            print_warning "Please set your OpenAI API key in the .env file"
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "OPENAI_API_KEY is configured"
        fi
    fi
}

check_required_config() {
    if [ -f "configuration.yml" ]; then
        if grep -q "api_key.*your_.*_api_key_here" configuration.yml || grep -q "api_key.*sk-example" configuration.yml || grep -q "api_key.*sk-test" configuration.yml; then
            print_warning "API key in configuration.yml appears to be a placeholder or example value"
            print_warning "Please set your actual API key in the configuration.yml file"
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "API key in configuration.yml appears to be configured"
        fi
        
        if grep -q "password.*password" configuration.yml && ! grep -q "password.*memmachine_password" configuration.yml; then
            print_warning "Database password in configuration.yml appears to be a placeholder"
            print_warning "Please set your actual database password in the configuration.yml file"
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "Database credentials in configuration.yml appear to be configured"
        fi
    fi
}

start_services() {
    print_info "Pulling and starting MemMachine services..."
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD up -d
    
    print_success "Services started successfully!"
}

wait_for_health() {
    print_info "Waiting for services to be healthy..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD ps
    
    print_info "Checking service health..."
    
    print_info "Waiting for PostgreSQL to be ready..."
    if run_with_timeout 120 "docker exec memmachine-postgres pg_isready -U ${POSTGRES_USER:-memmachine} -d ${POSTGRES_DB:-memmachine}"; then
        print_success "PostgreSQL is ready"
    else
        print_error "PostgreSQL failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
    
    print_info "Waiting for Neo4j to be ready..."
    if run_with_timeout 120 "docker exec memmachine-neo4j cypher-shell -u ${NEO4J_USER:-neo4j} -p ${NEO4J_PASSWORD:-neo4j_password} 'RETURN 1' > /dev/null 2>&1"; then
        print_success "Neo4j is ready"
    else
        print_error "Neo4j failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
    
    print_info "Waiting for local Ollama to be ready..."
    if run_with_timeout 120 "curl -f http://localhost:11434/api/tags > /dev/null 2>&1"; then
        print_success "Local Ollama is ready"
    else
        print_error "Local Ollama failed to become ready in 120 seconds. Please ensure Ollama is running locally."
        print_info "To start Ollama locally, run: ollama serve"
        exit 1
    fi
    
    print_info "Waiting for MemMachine to be ready..."
    if run_with_timeout 120 "curl -f http://localhost:${MEMORY_SERVER_PORT:-8080}/docs > /dev/null 2>&1"; then
        print_success "MemMachine is ready"
    else
        print_error "MemMachine failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
}

show_service_info() {
    print_success "MemMachine is now running!"
    echo ""
    echo "Service URLs:"
    echo "  MemMachine API: http://localhost:${MEMORY_SERVER_PORT:-8080}"
    echo "  Neo4j Browser: http://localhost:${NEO4J_HTTP_PORT:-7474}"
    echo "  Local Ollama API: http://localhost:11434"
    echo "  Health Check: http://localhost:${MEMORY_SERVER_PORT:-8080}/docs"
    echo "  Metrics: http://localhost:${MEMORY_SERVER_PORT:-8080}/metrics"
    echo ""
    echo "Database Access:"
    echo "  PostgreSQL: localhost:${POSTGRES_PORT:-5432} (user: ${POSTGRES_USER:-memmachine}, db: ${POSTGRES_DB:-memmachine})"
    echo "  Neo4j Bolt: localhost:${NEO4J_PORT:-7687} (user: ${NEO4J_USER:-neo4j})"
    echo ""
    echo "Useful Commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart: docker-compose restart"
    echo "  Clean up: docker-compose down -v"
    echo ""
}

main() {
    echo "MemMachine Docker Startup Script"
    echo "===================================="
    echo ""
    
    check_docker
    check_env_file
    check_config_file
    check_required_env
    check_required_config
    start_services
    wait_for_health
    show_service_info
}

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
    "help"|"-h"|"--help")
        echo "MemMachine Docker Startup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  Start MemMachine services"
        echo "  stop       Stop MemMachine services"
        echo "  restart    Restart MemMachine services"
        echo "  logs       Show service logs"
        echo "  clean      Remove all services and data"
        echo "  help       Show this help message"
        echo ""
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

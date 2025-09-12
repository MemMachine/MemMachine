#!/bin/bash

# MemMachine Docker Startup Script
# This script helps you get MemMachine running with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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
        if [ -f "env.example" ]; then
            cp env.example .env
            print_success "Created .env file from env.example"
            print_warning "Please edit .env file with your configuration before continuing"
            print_warning "Especially set your OPENAI_API_KEY"
            read -p "Press Enter to continue after editing .env file..."
        else
            print_error "env.example file not found. Please create .env file manually."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Check if required environment variables are set
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

# Build and start services
start_services() {
    print_status "Building and starting MemMachine services..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Build and start services
    $COMPOSE_CMD up -d --build
    
    print_success "Services started successfully!"
}

# Wait for services to be healthy
wait_for_health() {
    print_status "Waiting for services to be healthy..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Wait for services to be healthy
    $COMPOSE_CMD ps
    
    print_status "Checking service health..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL to be ready..."
    timeout 60 bash -c 'until docker exec memmachine-postgres pg_isready -U memmachine -d memmachine; do sleep 2; done'
    print_success "PostgreSQL is ready"
    
    # Wait for Neo4j
    print_status "Waiting for Neo4j to be ready..."
    timeout 60 bash -c 'until docker exec memmachine-neo4j cypher-shell -u neo4j -p neo4j_password "RETURN 1" > /dev/null 2>&1; do sleep 2; done'
    print_success "Neo4j is ready"
    
    # Wait for MemMachine
    print_status "Waiting for MemMachine to be ready..."
    timeout 120 bash -c 'until curl -f http://localhost:8080/health > /dev/null 2>&1; do sleep 5; done'
    print_success "MemMachine is ready"
}

# Show service information
show_service_info() {
    print_success "🎉 MemMachine is now running!"
    echo ""
    echo "Service URLs:"
    echo "  📊 MemMachine API: http://localhost:8080"
    echo "  🗄️  Neo4j Browser: http://localhost:7474"
    echo "  📈 Health Check: http://localhost:8080/health"
    echo "  📊 Metrics: http://localhost:8080/metrics"
    echo ""
    echo "Database Access:"
    echo "  🐘 PostgreSQL: localhost:5432 (user: memmachine, db: memmachine)"
    echo "  🔗 Neo4j Bolt: localhost:7687 (user: neo4j)"
    echo ""
    echo "Useful Commands:"
    echo "  📋 View logs: docker-compose logs -f"
    echo "  🛑 Stop services: docker-compose down"
    echo "  🔄 Restart: docker-compose restart"
    echo "  🧹 Clean up: docker-compose down -v"
    echo ""
}

# Main execution
main() {
    echo "MemMachine Docker Startup Script"
    echo "===================================="
    echo ""
    
    check_docker
    check_env_file
    check_required_env
    start_services
    wait_for_health
    show_service_info
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping MemMachine services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down
        else
            docker compose down
        fi
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting MemMachine services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose restart
        else
            docker compose restart
        fi
        print_success "Services restarted"
        ;;
    "logs")
        print_status "Showing MemMachine logs..."
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
            print_status "Cleaning up MemMachine services and data..."
            if command -v docker-compose &> /dev/null; then
                docker-compose down -v
            else
                docker compose down -v
            fi
            print_success "Cleanup completed"
        else
            print_status "Cleanup cancelled"
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

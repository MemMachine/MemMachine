#!/usr/bin/env bash

# MemMachine Docker Startup Script
# This script helps you get MemMachine running with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

is_first_run=false

# Use docker-compose or docker compose based on what's available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

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
        kill -0 "$cmd_pid" 2>/dev/null && kill -TERM "$cmd_pid" 2>/dev/null
    ) &

    local watchdog_pid=$!

    # Wait for the command to finish and suppress termination messages
    wait "$cmd_pid" 2>/dev/null
    local status=$?

    # Clean up watchdog if command finished early - suppress termination message
    kill -TERM "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

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
    echo -ne "${MAGENTA}[PROMPT]${NC} " >&2
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

# Function to escape special characters for sed replacement string
# In sed replacement strings, we only need to escape: & (ampersand) and \ (backslash)
escape_for_sed() {
    # Remove newlines and carriage returns first
    local cleaned=$(echo "$1" | tr -d '\n\r')
    # Escape backslashes first (must be done before escaping &)
    cleaned=$(echo "$cleaned" | sed 's/\\/\\\\/g')
    # Escape ampersands (used for matched text in sed replacement)
    echo "$cleaned" | sed 's/&/\\&/g'
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

# Prompt user for LLM model selection based on provider
select_llm_model() {
    local provider="$1"
    local llm_model=""
    
    case "$provider" in
        "OPENAI")
            print_prompt
            read -p "Which OpenAI LLM model would you like to use? [gpt-4o-mini]: " llm_model
            llm_model=$(echo "${llm_model:-gpt-4o-mini}" | tr -d '\n\r')
            print_success "Selected OpenAI LLM model: $llm_model" >&2
            ;;
        "BEDROCK")
            print_prompt
            read -p "Which AWS Bedrock LLM model would you like to use? [openai.gpt-oss-20b-1:0]: " llm_model
            llm_model=$(echo "${llm_model:-openai.gpt-oss-20b-1:0}" | tr -d '\n\r')
            print_success "Selected AWS Bedrock LLM model: $llm_model" >&2
            ;;
        "OLLAMA")
            print_prompt
            read -p "Which Ollama LLM model would you like to use? [llama3]: " llm_model
            llm_model=$(echo "${llm_model:-llama3}" | tr -d '\n\r')
            print_success "Selected Ollama LLM model: $llm_model" >&2
            ;;
        "OPENAI_COMPATIBLE")
            print_prompt
            read -p "Which OpenAI-compatible LLM model would you like to use? [qwen-flash]: " llm_model
            llm_model=$(echo "${llm_model:-qwen-flash}" | tr -d '\n\r')
            print_success "Selected OpenAI-compatible LLM model: $llm_model" >&2
            ;;
        *)
            print_warning "Unknown provider: $provider. Using default LLM model." >&2
            llm_model="gpt-4o-mini"
            ;;
    esac
    
    echo "$llm_model"
}

# Prompt user for embedding model selection based on provider
select_embedding_model() {
    local provider="$1"
    local embedding_model=""
    
    case "$provider" in
        "OPENAI")
            print_prompt
            read -p "Which OpenAI embedding model would you like to use? [text-embedding-3-small]: " embedding_model
            embedding_model=$(echo "${embedding_model:-text-embedding-3-small}" | tr -d '\n\r')
            print_success "Selected OpenAI embedding model: $embedding_model" >&2
            ;;
        "BEDROCK")
            print_prompt
            read -p "Which AWS Bedrock embedding model would you like to use? [amazon.titan-embed-text-v2:0]: " embedding_model
            embedding_model=$(echo "${embedding_model:-amazon.titan-embed-text-v2:0}" | tr -d '\n\r')
            print_success "Selected AWS Bedrock embedding model: $embedding_model" >&2
            ;;
        "OLLAMA")
            print_prompt
            read -p "Which Ollama embedding model would you like to use? [nomic-embed-text]: " embedding_model
            embedding_model=$(echo "${embedding_model:-nomic-embed-text}" | tr -d '\n\r')
            print_success "Selected Ollama embedding model: $embedding_model" >&2
            ;;
        "OPENAI_COMPATIBLE")
            print_prompt
            read -p "Which OpenAI-compatible embedding model would you like to use? [text-embedding-v4]: " embedding_model
            embedding_model=$(echo "${embedding_model:-text-embedding-v4}" | tr -d '\n\r')
            print_success "Selected OpenAI-compatible embedding model: $embedding_model" >&2
            ;;
        *)
            print_warning "Unknown provider: $provider. Using default embedding model." >&2
            embedding_model="text-embedding-3-small"
            ;;
    esac
    
    echo "$embedding_model"
}

# Generate configuration file with only the needed sections for the selected provider
generate_config_for_provider() {
    local config_source="$1"
    local provider="$2"
    local llm_model="$3"
    local embedding_model="$4"
    local escaped_llm_model=$(escape_for_sed "$llm_model")
    local escaped_embedding_model=$(escape_for_sed "$embedding_model")
    
    print_info "Generating configuration file for $provider provider..."
    
    # Determine which model and embedder to use based on provider
    case "$provider" in
        "OPENAI")
            local model_name="openai_model"
            local embedder_name="openai_embedder"
            local model_field="model"
            local embedder_field="model"
            ;;
        "BEDROCK")
            local model_name="aws_model"
            local embedder_name="aws_embedder_id"
            local model_field="model_id"
            local embedder_field="model_id"
            ;;
        "OLLAMA")
            local model_name="ollama_model"
            local embedder_name="ollama_embedder"
            local model_field="model"
            local embedder_field="model"
            ;;
        "OPENAI_COMPATIBLE")
            # OpenAI-compatible providers (e.g. self-hosted / compatible APIs)
            # Uses sample config entries: openai_compatible_model / openai_compatible_embedder
            local model_name="openai_compatible_model"
            local embedder_name="openai_compatible_embedder"
            local model_field="model"
            local embedder_field="model"
            ;;
        *)
            print_error "Unknown provider: $provider"
            return 1
            ;;
    esac
    
    # Use awk to extract and build the configuration file
    awk -v provider="$provider" \
        -v model_name="$model_name" \
        -v embedder_name="$embedder_name" \
        -v llm_model="$llm_model" \
        -v embedding_model="$embedding_model" \
        -v model_field="$model_field" \
        -v embedder_field="$embedder_field" \
        -f- "$config_source" <<'AWK_SCRIPT' > configuration.yml
    BEGIN {
        in_model_section = 0
        in_embedder_section = 0
        in_current_model = 0
        in_current_embedder = 0
        current_section = ""
        in_episodic = 0
        in_semantic = 0
        in_long_term = 0
        in_short_term = 0
    }
    
    # Track embedders and language_models sections (2 spaces, under resources:)
    /^  embedders:$/ {
        if (in_model_section || in_embedder_section) {
            print ""
        }
        in_embedder_section = 1
        in_model_section = 0
        in_current_embedder = 0
        print
        next
    }
    
    /^  language_models:$/ {
        if (in_model_section || in_embedder_section) {
            print ""
        }
        in_model_section = 1
        in_embedder_section = 0
        in_current_model = 0
        print
        next
    }
    
    # Exit embedders/language_models when hitting another 2-space section
    /^  [a-zA-Z_][a-zA-Z0-9_]*:$/ && !/^  (embedders|language_models):$/ && !in_episodic && !in_semantic {
        if (in_model_section || in_embedder_section) {
            print ""
            in_model_section = 0
            in_embedder_section = 0
        }
        print
        next
    }
    
    # Track current top-level section
    /^[a-zA-Z_][a-zA-Z0-9_]*:$/ && !/^  / {
        if (in_model_section || in_embedder_section) {
            print ""
        }
        current_section = substr($1, 1, length($1) - 1)
        in_model_section = 0
        in_embedder_section = 0
        in_current_model = 0
        in_current_embedder = 0
        
        # Track episodic_memory and semantic_memory sections
        if (current_section == "episodic_memory") {
            in_episodic = 1
            in_long_term = 0
            in_short_term = 0
        } else {
            in_episodic = 0
            in_long_term = 0
            in_short_term = 0
        }
        
        if (current_section == "semantic_memory") {
            in_semantic = 1
        } else {
            in_semantic = 0
        }
        
        print
        next
    }
    
    # Handle language_models section
    in_model_section {
        # Check if this is a model definition line (4 spaces)
        if (/^    [a-zA-Z_][a-zA-Z0-9_]*:$/) {
            model_key = substr($1, 1, length($1) - 1)  # Remove trailing :
            in_current_model = (model_key == model_name)
            print
            next
        }
        if (in_current_model) {
            # Replace model field value if this is the model line
            if (model_field == "model" && /^        model:/) {
                print "        model: \"" llm_model "\""
                next
            } else if (model_field == "model_id" && /^        model_id:/) {
                print "        model_id: \"" llm_model "\""
                next
            }
        }
        print
        next
    }
    
    # Handle embedders section
    in_embedder_section {
        # Check if this is an embedder definition line (4 spaces)
        if (/^    [a-zA-Z_][a-zA-Z0-9_]*:$/) {
            embedder_key = substr($1, 1, length($1) - 1)  # Remove trailing :
            in_current_embedder = (embedder_key == embedder_name)
            print
            next
        }
        if (in_current_embedder) {
            # Replace embedder model field value
            if (embedder_field == "model" && /^        model:/) {
                print "        model: \"" embedding_model "\""
                next
            } else if (embedder_field == "model_id" && /^        model_id:/) {
                print "        model_id: \"" embedding_model "\""
                next
            }
        }
        print
        next
    }
    
    # Handle episodic_memory section
    in_episodic {
        # Track long_term_memory subsection
        if (/^  long_term_memory:/) {
            in_long_term = 1
            in_short_term = 0
            print
            next
        }
        # Track short_term_memory subsection
        if (/^  short_term_memory:/) {
            in_short_term = 1
            in_long_term = 0
            print
            next
        }
        # Update embedder reference in long_term_memory
        if (in_long_term && /^    embedder:/) {
            print "    embedder: " embedder_name
            next
        }
        # Update llm_model reference in short_term_memory
        if (in_short_term && /^    llm_model:/) {
            print "    llm_model: " model_name
            next
        }
        print
        next
    }
    
    # Handle semantic_memory section - update model references
    in_semantic {
        if (/^  llm_model:/) {
            print "  llm_model: " model_name
            next
        } else if (/^  embedding_model:/) {
            print "  embedding_model: " embedder_name
            next
        }
        print
        next
    }
    
    # Default: print all other lines
    { print }
AWK_SCRIPT
    
    print_success "Generated configuration file with $provider provider settings"
}

# In lieu of yq, use awk to read over the configuration.yml file line-by-line,
# and set the database credentials using the same environment variables as in docker-compose.yml
set_config_defaults() {
    awk -v pg_user="${POSTGRES_USER:-memmachine}" \
        -v pg_pass="${POSTGRES_PASSWORD:-memmachine_password}" \
        -v pg_db="${POSTGRES_DB:-memmachine}" \
        -v neo4j_user="${NEO4J_USER:-neo4j}" \
        -v neo4j_pass="${NEO4J_PASSWORD:-neo4j_password}" '
/^storage:/ || /^vector_graph_store:/ {
  vendor = ""
}
/^[a-zA-Z][^:]*:/ && !/^storage:/ && !/^vector_graph_store:/ {
  vendor = ""
}

/vendor_name:/ {
  vendor = $2
  gsub(/^[ \t]+|[ \t]+$/, "", vendor)  # trim whitespace
}

/provider:/ && /neo4j/ {
  vendor = "neo4j"
}
/provider:/ && /postgres/ {
  vendor = "postgres"
}

vendor == "neo4j" && /host:/ { sub(/localhost/, "neo4j") }
vendor == "neo4j" && /uri:/ { sub(/localhost/, "neo4j") }
vendor == "neo4j" && /password:/ { sub(/<YOUR_PASSWORD_HERE>/, neo4j_pass) }

# Handle postgres configurations
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

        # Ask user for provider path (OpenAI, Bedrock, Ollama or OpenAI-compatible)
        print_prompt
        read -p "Which provider would you like to use? (OpenAI/Bedrock/Ollama/OpenAI-compatible) [OpenAI]: " provider_input
        # Clean the input and set default
        provider_input=$(echo "${provider_input:-OpenAI}" | tr -d '\n\r' | tr '[:lower:]' '[:upper:]' | tr '-' '_')
        local provider="$provider_input"
        
        # Validate provider selection
        if [[ "$provider" != "OPENAI" && "$provider" != "BEDROCK" && "$provider" != "OLLAMA" && "$provider" != "OPENAI_COMPATIBLE" ]]; then
            print_warning "Invalid provider selection: '$provider'. Defaulting to OpenAI."
            provider="OPENAI"
        fi
        
        print_info "Selected provider: $provider"

        # Update .env file with the selected image
        if [ -f ".env" ]; then
            # Remove existing MEMMACHINE_IMAGE from .env if it exists
            safe_sed_inplace '/^MEMMACHINE_IMAGE=/d' .env
        fi
        echo "MEMMACHINE_IMAGE=${MEMMACHINE_IMAGE}" >> .env
        print_success "Set MEMMACHINE_IMAGE to ${MEMMACHINE_IMAGE} in .env file"

        if [ -f "$CONFIG_SOURCE" ]; then
            # LLM model selection
            local selected_llm_model=$(select_llm_model "$provider")
            
            # embedding model selection
            local selected_embedding_model=$(select_embedding_model "$provider")
            
            # Generate configuration file with only needed sections for the selected provider
            generate_config_for_provider "$CONFIG_SOURCE" "$provider" "$selected_llm_model" "$selected_embedding_model"
        else
            print_error "$CONFIG_SOURCE file not found. Please create configuration.yml file manually."
            exit 1
        fi

        set_config_defaults
        is_first_run=true
    else
        print_success "configuration.yml file found"
    fi
}

select_openai_compatible_base_url() {
    local base_url=""
    local reply=""

    if [ "$is_first_run" = true ]; then
        print_prompt
        read -p "Model base URL is not set. Would you like to configure a custom model base URL? (y/N) " reply
        if [[ $reply =~ ^[Yy]$ ]]; then
            print_prompt
            read -p "OpenAI-compatible base URL [https://api.openai.com/v1]: " base_url
            base_url=$(echo "${base_url:-https://api.openai.com/v1}" | tr -d '\n\r')
            if [ -n "$base_url" ]; then
                # Update base_url under openai_compatible_model / openai_compatible_embedder.
                # Note: these entries exist in sample_configs/* and are included when provider=OPENAI_COMPATIBLE.
                safe_sed_inplace "/openai_compatible_model:/,/base_url:/ s|base_url: .*|base_url: \"$base_url\"|" configuration.yml
                safe_sed_inplace "/openai_compatible_embedder:/,/base_url:/ s|base_url: .*|base_url: \"$base_url\"|" configuration.yml
                print_success "Set OpenAI-compatible base URL to $base_url"
            fi
        fi
    else
        print_success "Model base URL appears to be configured"
    fi
}

# Prompt user if they would like to set their API keys based on provider; then set it in the .env file and configuration.yml file
set_provider_api_keys() {
    local api_key=""
    local aws_access_key=""
    local aws_secret_key=""
    local aws_region=""
    local model_id=""
    local base_url=""
    local reply=""
    
    if [ -f ".env" ]; then
        source .env
        
        # Get the actual model being used from configuration
        local llm_model=$(grep "llm_model:" configuration.yml | awk '{print $2}' | tr -d ' ')
        local embedder_model=$(grep "embedding_model:" configuration.yml | awk '{print $2}' | tr -d ' ')
        
        # Configure OpenAI if selected
        if [[ "$llm_model" == "openai_model" ]] || [[ "$embedder_model" == "openai_embedder" ]]; then
            if grep -q "<YOUR_API_KEY>" configuration.yml; then
                print_prompt
                read -p "OpenAI API key is not set. Would you like to set your OpenAI API key? (y/N) " reply
                if [[ $reply =~ ^[Yy]$ ]]; then
                    print_prompt
                    read -sp "Enter your OpenAI API key: " api_key
                    echo
                    # Use different delimiters to avoid conflicts with special characters
                    safe_sed_inplace "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" .env
                    safe_sed_inplace "s|api_key: <YOUR_API_KEY>|api_key: $api_key|g" configuration.yml
                    print_success "Set OPENAI_API_KEY in .env and configuration.yml"
                fi
            else
                print_success "OpenAI API key appears to be configured"
            fi
        fi

        # Configure OpenAI-compatible provider (OPENAI_COMPATIBLE)
        if [[ "$llm_model" == "openai_compatible_model" ]] || [[ "$embedder_model" == "openai_compatible_embedder" ]]; then
            if grep -q "<YOUR_API_KEY>" configuration.yml; then
                print_prompt
                read -p "API key is not set. Would you like to set your API key for the OpenAI-compatible provider? (y/N) " reply
                if [[ $reply =~ ^[Yy]$ ]]; then
                    print_prompt
                    read -sp "Enter your API key: " api_key
                    echo
                    safe_sed_inplace "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" .env
                    safe_sed_inplace "s|api_key: <YOUR_API_KEY>|api_key: $api_key|g" configuration.yml
                    print_success "Set OPENAI_API_KEY in .env and configuration.yml"
                fi
            else
                print_success "API key for OpenAI-compatible provider appears to be configured"
            fi

            # Base URL is configured only for OPENAI_COMPATIBLE.
            select_openai_compatible_base_url
        fi
        
        # Configure Bedrock if selected
        if [[ "$llm_model" == "aws_model" ]] || [[ "$embedder_model" == "aws_embedder_id" ]]; then
            if grep -q "<AWS_ACCESS_KEY_ID>" configuration.yml || grep -q "<AWS_SECRET_ACCESS_KEY>" configuration.yml; then
                print_prompt
                read -p "AWS credentials are not set. Would you like to set your AWS credentials for Bedrock? (y/N) " reply
                if [[ $reply =~ ^[Yy]$ ]]; then
                    print_prompt
                    read -sp "Enter your AWS Access Key ID: " aws_access_key
                    echo
                    print_prompt
                    read -sp "Enter your AWS Secret Access Key: " aws_secret_key
                    echo
                    print_prompt
                    read -p "Enter your AWS Region [us-west-2]: " aws_region
                    aws_region=${aws_region:-us-west-2}
                    
                    # Use different delimiters to avoid conflicts with special characters
                    safe_sed_inplace "s|aws_access_key_id: <AWS_ACCESS_KEY_ID>|aws_access_key_id: $aws_access_key|g" configuration.yml
                    safe_sed_inplace "s|aws_secret_access_key: <AWS_SECRET_ACCESS_KEY>|aws_secret_access_key: $aws_secret_key|g" configuration.yml
                    safe_sed_inplace "s|region: .*|region: \"$aws_region\"|g" configuration.yml
                    print_success "Set AWS credentials and configuration in configuration.yml"
                fi
            else
                print_success "AWS credentials appear to be configured"
            fi
        fi
        
        # Configure Ollama if selected
        if [[ "$llm_model" == "ollama_model" ]] || [[ "$embedder_model" == "ollama_embedder" ]]; then
            print_prompt
            read -p "Ollama base URL [http://host.docker.internal:11434/v1]: " base_url
            base_url=${base_url:-http://host.docker.internal:11434/v1}
            
            safe_sed_inplace "s|base_url: .*|base_url: \"$base_url\"|g" configuration.yml
            print_success "Set Ollama base URL: $base_url"
        fi
    fi
}

# Check if required environment variables are set
check_required_env() {
    if [ -f ".env" ]; then
        source .env
        
        # Get the actual model being used from configuration
        local llm_model=$(grep "llm_model:" configuration.yml | awk '{print $2}' | tr -d ' ')
        local embedder_model=$(grep "embedding_model:" configuration.yml | awk '{print $2}' | tr -d ' ')
        
        # Check OpenAI API key if OpenAI is configured
        if [[ "$llm_model" == "openai_model" ]] || [[ "$embedder_model" == "openai_embedder" ]]; then
            if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
                print_warning "OPENAI_API_KEY is not set or is using placeholder value"
                print_warning "Please set your OpenAI API key in the .env file"
                print_prompt
                read -p "Press Enter to continue anyway (some features may not work)..."
            else
                print_success "OPENAI_API_KEY is configured"
            fi
        fi

        # Check OpenAI-compatible provider API key if configured
        if [[ "$llm_model" == "openai_compatible_model" ]] || [[ "$embedder_model" == "openai_compatible_embedder" ]]; then
            if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
                print_warning "OPENAI_API_KEY is not set or is using placeholder value"
                print_warning "Please set your API key in the .env file for the OpenAI-compatible provider"
                print_prompt
                read -p "Press Enter to continue anyway (some features may not work)..."
            else
                print_success "OPENAI_API_KEY is configured (OpenAI-compatible provider)"
            fi

            if grep -q "openai_compatible_model:" configuration.yml && grep -q "base_url:" configuration.yml; then
                print_success "OpenAI-compatible base URL appears to be configured"
            else
                print_warning "OpenAI-compatible base URL may be missing in configuration.yml"
                print_prompt
                read -p "Press Enter to continue anyway (some features may not work)..."
            fi
        fi
        
        # Check AWS credentials if Bedrock is configured
        if [[ "$llm_model" == "aws_model" ]] || [[ "$embedder_model" == "aws_embedder_id" ]]; then
            if grep -q "<AWS_ACCESS_KEY_ID>" configuration.yml || grep -q "<AWS_SECRET_ACCESS_KEY>" configuration.yml; then
                print_warning "AWS credentials are not set or are using placeholder values"
                print_warning "Please set your AWS credentials in the configuration.yml file"
                print_prompt
                read -p "Press Enter to continue anyway (some features may not work)..."
            else
                print_success "AWS credentials appear to be configured"
            fi
        fi
        
        # Check Ollama configuration
        if [[ "$llm_model" == "ollama_model" ]] || [[ "$embedder_model" == "ollama_embedder" ]]; then
            if grep -q "base_url: \"http://host.docker.internal:11434/v1\"" configuration.yml; then
                print_success "Ollama configuration detected with default base URL"
            else
                print_success "Ollama configuration detected with custom base URL"
            fi
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
    if timeout 120 bash -c "until curl -f http://localhost:${MEMORY_SERVER_PORT:-8080}/api/v2/health > /dev/null 2>&1; do sleep 5; done"; then
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
    echo "  📊 MemMachine API Docs: http://localhost:${MEMORY_SERVER_PORT:-8080}/docs"
    echo "  🗄️  Neo4j Browser: http://localhost:${NEO4J_HTTP_PORT:-7474}"
    echo "  📈 Health Check: http://localhost:${MEMORY_SERVER_PORT:-8080}/api/v2/health"
    echo "  📊 Metrics: http://localhost:${MEMORY_SERVER_PORT:-8080}/api/v2/metrics"
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
    local gpu="false" # default to false
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
        read -p "Building $name with '--build-arg GPU=[true|false]' (default: false): " reply
        gpu=$(echo "${reply:-false}" | tr '[:upper:]' '[:lower:]')
        if [[ "$gpu" != "true" && "$gpu" != "false" ]]; then
            print_error "Invalid value for GPU: $gpu"
            exit 1
        fi
    else
        print_info "Building $name with '--build-arg GPU=$gpu'"
    fi

    # Proceed with build after validation passes
    print_info "Building $name with '--build-arg GPU=$gpu'"
    docker build --build-arg GPU=$gpu -t "$name" .
}

# Main execution
main() {
    echo "MemMachine Docker Startup Script"
    echo "===================================="
    echo ""
    
    check_docker
    check_env_file
    check_config_file
    set_provider_api_keys
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
        $COMPOSE_CMD down
        print_success "Services stopped"
        ;;
    "restart")
        print_info "Restarting MemMachine services..."
        $COMPOSE_CMD restart
        print_success "Services restarted"
        ;;
    "logs")
        print_info "Showing MemMachine logs..."
        $COMPOSE_CMD logs -f
        ;;
    "clean")
        print_warning "This will remove all data and volumes!"
        print_prompt
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Cleaning up MemMachine services and data..."
            $COMPOSE_CMD down -v
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
        echo "Provider Options:"
        echo "  OpenAI    - Uses OpenAI's GPT models and embedding models"
        echo "             Default LLM: gpt-4o-mini"
        echo "             Default embedding: text-embedding-3-small"
        echo "             Requires: OpenAI API key"
        echo "  Bedrock   - Uses AWS Bedrock models"
        echo "             Default LLM: openai.gpt-oss-20b-1:0"
        echo "             Default embedding: amazon.titan-embed-text-v2:0"
        echo "             Requires: AWS Access Key ID, Secret Key, Region, Model ID"
        echo "  Ollama    - Uses local Ollama models"
        echo "             Default LLM: llama3"
        echo "             Default embedding: nomic-embed-text"
        echo "             Requires: Base URL (default: http://host.docker.internal:11434/v1)"
        echo "  OpenAI-compatible - Uses an OpenAI-compatible endpoint (custom base URL)"
        echo "             Default LLM: qwen-flash"
        echo "             Default embedding: text-embedding-v4"
        echo "             Requires: API key (OPENAI_API_KEY) and base URL"
        echo ""
        echo "Features:"
        echo "  ProfileMemory - Intelligent user profiling and memory management"
        echo "  Episodic Memory - Context-aware memory storage and retrieval"
        echo "  Multi-provider support - Choose your preferred AI provider"
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


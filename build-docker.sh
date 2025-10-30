#!/usr/bin/env bash
#
# Description:
# Builds, tests, and pushes multi-architecture Docker images for the MemMachine project.
# The script can be run interactively or with command-line flags for automation.
#
# Interactive Usage:
#   ./build-local.sh
#
# Push-Only Mode:
#   ./build-local.sh --push-only
#
# Non-Interactive Build Example:
#   ./build-local.sh --type gpu --arch both --version v0.1.1 --push --user myuser --token mytoken
#

# --- Strict Mode ---
set -euo pipefail

# --- Configuration ---
readonly SCRIPT_NAME="build-docker.sh"
readonly IMAGE_NAME="memmachine/memmachine"
readonly BUILDER_NAME="memmachine_builder"

# --- Helper Functions for Colored Output ---
if tput setaf 1 >&/dev/null; then
    readonly COLOR_RESET="$(tput sgr0)"
    readonly COLOR_RED="$(tput setaf 1)"
    readonly COLOR_GREEN="$(tput setaf 2)"
    readonly COLOR_YELLOW="$(tput setaf 3)"
    readonly COLOR_BLUE="$(tput setaf 4)"
else
    readonly COLOR_RESET=""
    readonly COLOR_RED=""
    readonly COLOR_GREEN=""
    readonly COLOR_YELLOW=""
    readonly COLOR_BLUE=""
fi

msg() { echo "${COLOR_BLUE}==>${COLOR_RESET} ${COLOR_YELLOW}$1${COLOR_RESET}"; }
success() { echo "${COLOR_BLUE}==>${COLOR_RESET} ${COLOR_GREEN}$1${COLOR_RESET}"; }
error() { echo "${COLOR_RED}ERROR:${COLOR_RESET} $1" >&2; exit 1; }

# --- Script Functions ---

show_usage() {
    echo "Usage: ./${SCRIPT_NAME} [options]"
    echo ""
    echo "Actions:"
    echo "  --push-only               Detect local images and prompt to push them."
    echo ""
    echo "Build Options (ignored if --push-only is used):"
    echo "  --type <cpu|gpu|both>   Specify the image type to build."
    echo "  --arch <amd64|arm64|both> Specify the architecture."
    echo "  --version <tag>           Set the version tag (e.g., v0.1.1)."
    echo "  --latest                  Apply 'latest' tags to the build."
    echo "  --push                    Build and push. If omitted, builds locally."
    echo ""
    echo "Credential Options:"
    echo "  --user <username>         Docker Hub username for non-interactive login."
    echo "  --token <token>           Docker Hub token for non-interactive login."
    echo ""
    echo "Other Options:"
    echo "  -h, --help                Show this help message."
    echo ""
    echo "If no options are provided, the script will run in interactive build mode."
}

check_dependencies() {
    msg "Checking for required dependencies..."
    if ! command -v git &> /dev/null; then error "Git is not installed."; fi
    if ! command -v docker &> /dev/null; then error "Docker is not installed."; fi
    if [[ "$ACTION_MODE" != "push-only" && ! -f "Dockerfile" ]]; then
        error "Dockerfile not found. Please run this script from the project's root directory."
    fi
    success "All dependencies are present."
}

setup_buildx() {
    msg "Setting up Docker Buildx multi-architecture builder..."
    if ! docker buildx ls | grep -q "${BUILDER_NAME}.*running"; then
        msg "Builder '${BUILDER_NAME}' not found or not running. Creating it now..."
        docker buildx rm "${BUILDER_NAME}" &>/dev/null || true
        docker buildx create --name "${BUILDER_NAME}" --use --driver-opt "image=moby/buildkit:v0.12.0" || error "Failed to create Buildx builder."
    fi
    docker buildx inspect --bootstrap || error "Failed to bootstrap Buildx builder."
    success "Buildx builder is ready."
}

handle_docker_login() {
    msg "Authenticating with Docker Hub..."
    if [[ -n "${DOCKER_USER:-}" && -n "${DOCKER_TOKEN:-}" ]]; then
        msg "Using credentials from command-line arguments."
        if ! echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin; then
            error "Docker login failed with provided credentials."
        fi
    else
        msg "Please enter your Docker Hub credentials."
        read -p "Username: " docker_user_interactive
        read -sp "Token or Password: " docker_token_interactive; echo ""
        if ! echo "$docker_token_interactive" | docker login -u "$docker_user_interactive" --password-stdin; then
            error "Docker login failed. Please check your credentials."
        fi
    fi
    success "Docker login successful."
}

# --- Global variables for build choices ---
ACTION_MODE="build"
BUILD_CPU=false
BUILD_GPU=false
PLATFORMS=""
VERSION=""
PUSH_LATEST=false
ACTION="Local Build"
INTERACTIVE=true
DOCKER_USER=""
DOCKER_TOKEN=""

parse_args() {
    if [[ $# -gt 0 ]]; then INTERACTIVE=false; fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --push-only) ACTION_MODE="push-only"; shift 1 ;;
            --type) BUILD_TYPE="$2"; shift 2 ;;
            --arch) ARCH_TYPE="$2"; shift 2 ;;
            --version) VERSION="$2"; shift 2 ;;
            --latest) PUSH_LATEST=true; shift 1 ;;
            --push) ACTION="Build and Push"; shift 1 ;;
            --user) DOCKER_USER="$2"; shift 2 ;;
            --token) DOCKER_TOKEN="$2"; shift 2 ;;
            -h|--help) show_usage; exit 0 ;;
            *) error "Unknown option: $1";;
        esac
    done

    if [[ "$ACTION_MODE" == "build" && "$INTERACTIVE" == "false" ]]; then
        # Validate and set build type
        case "${BUILD_TYPE:-}" in
            cpu) BUILD_CPU=true ;; gpu) BUILD_GPU=true ;; both) BUILD_CPU=true; BUILD_GPU=true ;;
            "") error "--type is a required argument for non-interactive mode.";;
            *) error "Invalid value for --type: ${BUILD_TYPE}";;
        esac
        # Validate and set architecture
        case "${ARCH_TYPE:-}" in
            amd64) PLATFORMS="linux/amd64" ;; arm64) PLATFORMS="linux/arm64" ;; both) PLATFORMS="linux/amd64,linux/arm64" ;;
            "") error "--arch is a required argument for non-interactive mode.";;
            *) error "Invalid value for --arch: ${ARCH_TYPE}";;
        esac
        # Validate version
        if [[ -z "${VERSION:-}" ]]; then error "--version is a required argument for non-interactive mode."; fi
    fi
}

get_interactive_choices() {
    msg "Select build options:"
    read -p "Which image type to build? (1: CPU, 2: GPU, 3: Both) [3]: " build_choice
    case "${build_choice:-3}" in
        1) BUILD_CPU=true ;; 2) BUILD_GPU=true ;; 3) BUILD_CPU=true; BUILD_GPU=true ;;
        *) error "Invalid build type choice." ;;
    esac

    read -p "Which architectures? (1: amd64, 2: arm64, 3: Both) [3]: " arch_choice
    case "${arch_choice:-3}" in
        1) PLATFORMS="linux/amd64" ;; 2) PLATFORMS="linux/arm64" ;; 3) PLATFORMS="linux/amd64,linux/arm64" ;;
        *) error "Invalid architecture choice." ;;
    esac

    local latest_git_tag; latest_git_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "none")
    if [[ "$latest_git_tag" != "none" ]]; then
        read -p "Latest Git tag is '${latest_git_tag}'. Use this? (Y/n): " use_latest_tag
        if [[ ! "$use_latest_tag" =~ ^[nN]$ ]]; then VERSION="$latest_git_tag"; fi
    fi
    if [[ -z "$VERSION" ]]; then
        read -p "Please enter the version tag (e.g., v0.1.1): " VERSION
        if [[ -z "$VERSION" ]]; then error "Version tag cannot be empty."; fi
    fi

    msg "A 'latest' tag (e.g., 'latest-gpu') acts as a pointer to the newest stable release."
    read -p "Apply 'latest' tags for this version? (Y/n): " push_latest_choice
    if [[ ! "$push_latest_choice" =~ ^[nN]$ ]]; then PUSH_LATEST=true; fi

    read -p "Select action (1: Build locally, 2: Build and Push, 3: Cancel) [2]: " action_choice
    case "${action_choice:-2}" in
        1) ACTION="Local Build" ;; 2) ACTION="Build and Push" ;; 3) msg "Operation cancelled."; exit 0 ;;
        *) error "Invalid action choice." ;;
    esac
}

print_summary_and_confirm() {
    local build_types=""
    if [[ "$BUILD_CPU" == true ]]; then build_types="CPU "; fi
    if [[ "$BUILD_GPU" == true ]]; then build_types+="GPU"; fi
    
    msg "--- BUILD SUMMARY ---"
    echo -e "Image Types: \t ${build_types}"
    echo -e "Version: \t\t ${VERSION}"
    echo -e "Architectures: \t ${PLATFORMS}"
    echo -e "Apply 'latest' tags: \t ${PUSH_LATEST}"
    echo -e "Action: \t\t ${ACTION}"
    echo -e "---------------------"

    if [[ "$INTERACTIVE" == true ]]; then
        read -p "Continue? (Y/n): " confirm
        if [[ "$confirm" =~ ^[nN]$ ]]; then msg "Operation cancelled by user."; exit 0; fi
    fi
}

build_image() {
    local build_type="$1"
    local build_type_upper=$(echo "$build_type" | tr '[:lower:]' '[:upper:]')
    msg "Preparing to build ${build_type_upper} image..."
    local build_args=""; if [[ "$build_type" == "gpu" ]]; then build_args="--build-arg GPU=true"; fi

    local docker_tags=(); docker_tags+=("--tag" "${IMAGE_NAME}:${VERSION}-${build_type}")
    if [[ "$PUSH_LATEST" == "true" ]]; then
        docker_tags+=("--tag" "${IMAGE_NAME}:latest-${build_type}")
        if [[ "$build_type" == "cpu" ]]; then docker_tags+=("--tag" "${IMAGE_NAME}:latest"); fi
    fi
    
    local final_args=()
    if [[ "$ACTION" == "Local Build" ]]; then
        if [[ "$PLATFORMS" == *,* ]]; then error "Cannot load a multi-platform build locally. Choose a single architecture or select 'Build and Push'."; fi
        final_args+=("--load")
    else
        final_args+=("--push")
    fi

    # shellcheck disable=SC2086
    docker buildx build --platform "${PLATFORMS}" ${build_args} "${docker_tags[@]}" "${final_args[@]}" . || error "Docker build for ${build_type_upper} failed."

    if [[ "$ACTION" == "Local Build" ]]; then
        success "${build_type_upper} image built successfully!"
        msg "To test it, run: docker run -it --rm ${IMAGE_NAME}:${VERSION}-${build_type} bash"
    else
        success "${build_type_upper} image build and push completed successfully!"
    fi
}

push_only_mode() {
    msg "Detecting locally built '${IMAGE_NAME}' images..."
    local images_found=()
    # Read all local images matching the name into an array
    while IFS= read -r line; do
        images_found+=("$line")
    done < <(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^${IMAGE_NAME}:" || true)

    if [[ ${#images_found[@]} -eq 0 ]]; then
        error "No local images found for repository '${IMAGE_NAME}'. Please build them first."
    fi

    msg "Found the following images:"
    for i in "${!images_found[@]}"; do
        echo "  $((i+1))) ${images_found[$i]}"
    done

    read -p "Enter the numbers of the images to push (e.g., 1,3 or 'all'): " selection
    if [[ -z "$selection" ]]; then msg "No selection made. Exiting."; exit 0; fi

    local images_to_push=()
    if [[ "$selection" == "all" ]]; then
        images_to_push=("${images_found[@]}")
    else
        # Split the comma-separated string into an array of numbers
        IFS=',' read -ra selected_indices <<< "$selection"
        for index in "${selected_indices[@]}"; do
            # Validate that the index is a number and within bounds
            if [[ "$index" =~ ^[0-9]+$ && $((index-1)) -ge 0 && $((index-1)) -lt ${#images_found[@]} ]]; then
                images_to_push+=("${images_found[$((index-1))]}")
            else
                error "Invalid selection: '${index}'. Please enter a valid number from the list."
            fi
        done
    fi

    if [[ ${#images_to_push[@]} -eq 0 ]]; then msg "No valid images selected. Exiting."; exit 0; fi

    msg "The following images will be pushed:"
    for image in "${images_to_push[@]}"; do
        echo "  - ${image}"
    done
    read -p "Continue? (Y/n): " confirm
    if [[ "$confirm" =~ ^[nN]$ ]]; then msg "Push cancelled."; exit 0; fi

    handle_docker_login

    for image in "${images_to_push[@]}"; do
        msg "Pushing ${image}..."
        docker push "${image}" || error "Failed to push ${image}."
        success "Successfully pushed ${image}."
    done
}

# --- Main Script Logic ---
main() {
    parse_args "$@"

    if [[ "$ACTION_MODE" == "push-only" ]]; then
        push_only_mode
    else
        if [[ "$INTERACTIVE" == true ]]; then get_interactive_choices; fi
        check_dependencies
        setup_buildx
        print_summary_and_confirm
        if [[ "$ACTION" == "Build and Push" ]]; then handle_docker_login; fi
        if [[ "$BUILD_CPU" == true ]]; then build_image "cpu"; fi
        if [[ "$BUILD_GPU" == true ]]; then build_image "gpu"; fi
    fi
    success "All selected operations are complete."
}

# Run the main function, passing all command-line arguments to it
main "$@"

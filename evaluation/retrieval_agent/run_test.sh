#!/usr/bin/env bash

usage_locomo() {
    echo "Locomo Usage: $0 locomo RESULT_POSTFIX RUN_TYPE TEST_TARGET [options]"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [memmachine | retrieval_agent | llm]"
    echo "Options:"
    echo "  --ingest-concurrency N"
    echo "                     Optional max concurrent LoCoMo ingestion tasks"
    echo "                     (ingest only, default: 10)"
    echo "  --search-concurrency N"
    echo "                     Optional max concurrent LoCoMo search requests"
    echo "                     (search only, default: 1)"
    echo "  --judge-concurrency N"
    echo "                     Optional max concurrent LLM judge workers"
    echo "                     (search only, default: 30)"
    exit 1
}

usage_wiki() {
    echo "WikiMultihop Usage: wikimultihop $0 RESULT_POSTFIX RUN_TYPE TEST_TARGET LENGTH"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [memmachine | retrieval_agent | llm]"
    echo "  LENGTH            Number of examples to run [1 - 12576]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Optional max concurrent WikiMultiHop search requests"
    echo "                     (search only, default: 10)"
    echo "  --judge-concurrency N"
    echo "                     Optional max concurrent LLM judge workers"
    echo "                     (search only, default: 30)"
    exit 1
}

usage_hotpotqa() {
    echo "HotpotQA Usage: $0 hotpotqa RESULT_POSTFIX RUN_TYPE SPLIT_NAME TEST_TARGET LENGTH"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  SPLIT_NAME        Dataset split name [train | validation]. Train set contains 19.9%"
    echo "                      easy, 62.8% medium, 17.3% hard questions. Validation set contains"
    echo "                      hard questions only."
    echo "  TEST_TARGET       [memmachine | retrieval_agent | llm]"
    echo "  LENGTH            Number of examples to run [train set 1 - 90447 | validation set 1 - 7405]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Optional max concurrent HotpotQA search requests"
    echo "                     (search only, default: 30)"
    echo "  --judge-concurrency N"
    echo "                     Optional max concurrent LLM judge workers"
    echo "                     (search only, default: 30)"
    exit 1
}

usage_longmemeval() {
    echo "LongMemEval Usage: $0 longmemeval RESULT_POSTFIX RUN_TYPE SPLIT_NAME TEST_TARGET LENGTH"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  SPLIT_NAME        Dataset split name, e.g. longmemeval_s_cleaned"
    echo "  TEST_TARGET       [memmachine | retrieval_agent | llm]"
    echo "  LENGTH            Number of examples to run [1 - split size]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Optional max concurrent LongMemEval search requests"
    echo "                     (search only, default: 30)"
    echo "  --judge-concurrency N"
    echo "                     Optional max concurrent LLM judge workers"
    echo "                     (search only, default: 30)"
    exit 1
}

usage_beam() {
    echo "BEAM Usage: $0 beam RESULT_POSTFIX RUN_TYPE TEST_TARGET CHAT_PATH QUESTIONS_PATH"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [memmachine | retrieval_agent | llm]"
    echo "  CHAT_PATH         Path to chat.json file"
    echo "  QUESTIONS_PATH    Path to probing_questions.json file"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Optional max concurrent BEAM search requests"
    echo "                     (search only, default: 10)"
    echo "  --judge-concurrency N"
    echo "                     Optional max concurrent LLM judge workers"
    echo "                     (search only, default: 30)"
    exit 1
}

show_help() {
    case "$1" in
        locomo)
            usage_locomo
            ;;
        wikimultihop)
            usage_wiki
            ;;
        hotpotqa)
            usage_hotpotqa
            ;;
        longmemeval)
            usage_longmemeval
            ;;
        beam)
            usage_beam
            ;;
        ""|all)
            echo "Usage: $0 TEST [args...]"
            echo
            echo "Available TEST values:"
            echo "  locomo"
            echo "  wikimultihop"
            echo "  hotpotqa"
            echo "  longmemeval"
            echo "  beam"
            echo
            echo "Use:"
            echo "  $0 TEST --help"
            echo "to see test-specific usage."
            exit 0
            ;;
        *)
            echo "Unknown test: $1"
            show_help all
            ;;
    esac
}

POSITIONAL_ARGS=()
INGEST_CONCURRENCY=""
SEARCH_CONCURRENCY=""
JUDGE_CONCURRENCY=""
PYTHON_CMD=(python)
PYTHON_INSTALL_CMD='python -m pip install -r requirements.txt'

parse_optional_flags() {
    POSITIONAL_ARGS=()
    INGEST_CONCURRENCY=""
    SEARCH_CONCURRENCY=""
    JUDGE_CONCURRENCY=""

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --ingest-concurrency)
                if [ "$#" -lt 2 ]; then
                    echo "Error: --ingest-concurrency requires a value"
                    exit 1
                fi
                INGEST_CONCURRENCY="$2"
                shift 2
                ;;
            --ingest-concurrency=*)
                INGEST_CONCURRENCY="${1#*=}"
                shift
                ;;
            --search-concurrency)
                if [ "$#" -lt 2 ]; then
                    echo "Error: --search-concurrency requires a value"
                    exit 1
                fi
                SEARCH_CONCURRENCY="$2"
                shift 2
                ;;
            --search-concurrency=*)
                SEARCH_CONCURRENCY="${1#*=}"
                shift
                ;;
            --judge-concurrency)
                if [ "$#" -lt 2 ]; then
                    echo "Error: --judge-concurrency requires a value"
                    exit 1
                fi
                JUDGE_CONCURRENCY="$2"
                shift 2
                ;;
            --judge-concurrency=*)
                JUDGE_CONCURRENCY="${1#*=}"
                shift
                ;;
            *)
                POSITIONAL_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

validate_args() {
    case "$1" in
        locomo)
            if [ "$#" -ne 4 ]; then
                show_help locomo
            fi
            if [ -n "${INGEST_CONCURRENCY:-}" ] && [ "$3" != "ingest" ]; then
                echo "--ingest-concurrency can only be used with locomo ingest"
                echo
                show_help locomo
            fi
            if [ -n "${SEARCH_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--search-concurrency can only be used with search runs"
                echo
                show_help locomo
            fi
            if [ -n "${JUDGE_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--judge-concurrency can only be used with search runs"
                echo
                show_help locomo
            fi
            ;;
        wikimultihop)
            if [ -n "${INGEST_CONCURRENCY:-}" ]; then
                echo "--ingest-concurrency is only supported for locomo ingest"
                exit 1
            fi
            if [ "$#" -ne 5 ]; then
                show_help wikimultihop
            fi
            if [ -n "${SEARCH_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--search-concurrency can only be used with search runs"
                echo
                show_help wikimultihop
            fi
            if [ -n "${JUDGE_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--judge-concurrency can only be used with search runs"
                echo
                show_help wikimultihop
            fi
            ;;
        hotpotqa)
            if [ -n "${INGEST_CONCURRENCY:-}" ]; then
                echo "--ingest-concurrency is only supported for locomo ingest"
                exit 1
            fi
            if [ "$#" -ne 6 ]; then
                show_help hotpotqa
            fi
            if [ -n "${SEARCH_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--search-concurrency can only be used with search runs"
                echo
                show_help hotpotqa
            fi
            if [ -n "${JUDGE_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--judge-concurrency can only be used with search runs"
                echo
                show_help hotpotqa
            fi
            ;;
        longmemeval)
            if [ -n "${INGEST_CONCURRENCY:-}" ]; then
                echo "--ingest-concurrency is only supported for locomo ingest"
                exit 1
            fi
            if [ "$#" -ne 6 ]; then
                show_help longmemeval
            fi
            if [ -n "${SEARCH_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--search-concurrency can only be used with search runs"
                echo
                show_help longmemeval
            fi
            if [ -n "${JUDGE_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--judge-concurrency can only be used with search runs"
                echo
                show_help longmemeval
            fi
            ;;
        beam)
            if [ -n "${INGEST_CONCURRENCY:-}" ]; then
                echo "--ingest-concurrency is only supported for locomo ingest"
                exit 1
            fi
            if [ "$#" -ne 6 ]; then
                show_help beam
            fi
            if [ -n "${SEARCH_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--search-concurrency can only be used with search runs"
                echo
                show_help beam
            fi
            if [ -n "${JUDGE_CONCURRENCY:-}" ] && [ "$3" != "search" ]; then
                echo "--judge-concurrency can only be used with search runs"
                echo
                show_help beam
            fi
            ;;
        *)
            echo "Unknown test: $TEST"
            show_help all
            ;;
    esac
}

validate_positive_integer() {
    case "$1" in
        ''|*[!0-9]*)
            return 1
            ;;
        0)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

check_python_modules() {
    if ! "${PYTHON_CMD[@]}" "$SCRIPT_DIR/preflight.py" "$@"; then
        echo "Install benchmark dependencies with:"
        echo "  (cd \"$SCRIPT_DIR\" && ${PYTHON_INSTALL_CMD})"
        return 1
    fi
}

run_test() {
    TEST="$1"

    if [ -n "${INGEST_CONCURRENCY:-}" ] && ! validate_positive_integer "$INGEST_CONCURRENCY"; then
        echo "--ingest-concurrency must be a positive integer"
        exit 1
    fi
    if [ -n "${SEARCH_CONCURRENCY:-}" ] && ! validate_positive_integer "$SEARCH_CONCURRENCY"; then
        echo "--search-concurrency must be a positive integer"
        exit 1
    fi
    if [ -n "${JUDGE_CONCURRENCY:-}" ] && ! validate_positive_integer "$JUDGE_CONCURRENCY"; then
        echo "--judge-concurrency must be a positive integer"
        exit 1
    fi

    PYTHON_CMD=(python)
    PYTHON_INSTALL_CMD='python -m pip install -r requirements.txt'

    case "$TEST" in
        locomo)
            RESULT_POSTFIX=$2
            INGEST=$3
            TEST_TARGET=$4
            ;;
        wikimultihop)
            RESULT_POSTFIX=$2
            INGEST=$3
            TEST_TARGET=$4
            LENGTH=$5
            ;;
        hotpotqa)
            RESULT_POSTFIX=$2
            INGEST=$3
            SPLIT_NAME=$4
            TEST_TARGET=$5
            LENGTH=$6
            ;;
        longmemeval)
            RESULT_POSTFIX=$2
            INGEST=$3
            SPLIT_NAME=$4
            TEST_TARGET=$5
            LENGTH=$6
            ;;
        beam)
            RESULT_POSTFIX=$2
            INGEST=$3
            TEST_TARGET=$4
            CHAT_PATH=$5
            QUESTIONS_PATH=$6
            ;;
        *)
            echo "Unknown test: $TEST"
            show_help all
            ;;
    esac

    SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
    CONFIG_FILE="${SCRIPT_DIR}/configuration.yml"

    # Require configuration.yml in the same directory as this script
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: configuration.yml not found at '${CONFIG_FILE}'"
        echo
        echo "Please create a configuration.yml in the retrieval_agent directory before"
        echo "running benchmarks. See evaluation/retrieval_agent/README.md for details"
        echo "and configuration samples."
        exit 1
    fi

    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/common/src:${REPO_ROOT}/packages/server/src:${REPO_ROOT}/packages/client/src${PYTHONPATH:+:${PYTHONPATH}}"
    mkdir -p "${SCRIPT_DIR}/result/final_score"
    RESULT_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_output_${RESULT_POSTFIX}.json"
    EVAL_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_evaluation_metrics_${RESULT_POSTFIX}.json"
    FINAL_SCORE_FILE="${SCRIPT_DIR}/result/final_score/${TEST}_${TEST_TARGET}_${RESULT_POSTFIX}.result"
    SESSION_ID="${TEST}_${RESULT_POSTFIX}"

    rm -f "$RESULT_FILE" "$EVAL_FILE" "$FINAL_SCORE_FILE"

    case "$TEST" in
        locomo)
            INGEST_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/locomo_ingest.py" --data-path "$SCRIPT_DIR/../data/locomo10.json" --config-path "$CONFIG_FILE")
            SEARCH_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/locomo_search.py" --data-path "$SCRIPT_DIR/../data/locomo10.json" --eval-result-path "$RESULT_FILE" --test-target "$TEST_TARGET" --config-path "$CONFIG_FILE")
            if [ -n "${INGEST_CONCURRENCY:-}" ]; then
                INGEST_CMD+=(--concurrency "$INGEST_CONCURRENCY")
            fi
            if [ -n "${SEARCH_CONCURRENCY:-}" ]; then
                SEARCH_CMD+=(--concurrency "$SEARCH_CONCURRENCY")
            fi
            ;;
        wikimultihop)
            INGEST_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/wikimultihop_ingest.py" --data-path "$SCRIPT_DIR/../data/wikimultihop.json" --length "$LENGTH" --config-path "$CONFIG_FILE")
            SEARCH_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/wikimultihop_search.py" --data-path "$SCRIPT_DIR/../data/wikimultihop.json" --eval-result-path "$RESULT_FILE" --test-target "$TEST_TARGET" --length "$LENGTH" --config-path "$CONFIG_FILE")
            if [ -n "${SEARCH_CONCURRENCY:-}" ]; then
                SEARCH_CMD+=(--concurrency "$SEARCH_CONCURRENCY")
            fi
            ;;
        hotpotqa)
            INGEST_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/hotpotQA_test.py" --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --config-path "$CONFIG_FILE")
            SEARCH_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/hotpotQA_test.py" --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --config-path "$CONFIG_FILE")
            if [ -n "${SEARCH_CONCURRENCY:-}" ]; then
                SEARCH_CMD+=(--concurrency "$SEARCH_CONCURRENCY")
            fi
            ;;
        longmemeval)
            PYTHON_CMD=(uv run python)
            PYTHON_INSTALL_CMD='uv run python -m pip install -r requirements.txt'
            INGEST_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/longmemeval_test.py" --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID" --config-path "$CONFIG_FILE")
            SEARCH_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/longmemeval_test.py" --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID" --config-path "$CONFIG_FILE")
            if [ -n "${SEARCH_CONCURRENCY:-}" ]; then
                SEARCH_CMD+=(--concurrency "$SEARCH_CONCURRENCY")
            fi
            ;;
        beam)
            INGEST_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/beam_ingest.py" --data-path "$CHAT_PATH" --config-path "$CONFIG_FILE" --session-id "$SESSION_ID")
            SEARCH_CMD=("${PYTHON_CMD[@]}" -u "$SCRIPT_DIR/beam_search.py" --chat-data-path "$CHAT_PATH" --data-path "$QUESTIONS_PATH" --eval-result-path "$RESULT_FILE" --config-path "$CONFIG_FILE" --session-id "$SESSION_ID" --test-target "$TEST_TARGET")
            EVALUATE_CMD=("${PYTHON_CMD[@]}" "$SCRIPT_DIR/beam_evaluate.py" --data-path "$RESULT_FILE" --target-path "$EVAL_FILE" --config-path "$CONFIG_FILE")
            if [ -n "${SEARCH_CONCURRENCY:-}" ]; then
                SEARCH_CMD+=(--concurrency "$SEARCH_CONCURRENCY")
            fi
            if [ -n "${JUDGE_CONCURRENCY:-}" ]; then
                EVALUATE_CMD+=(--max-workers "$JUDGE_CONCURRENCY")
            fi
            ;;
    esac

    if [[ "$INGEST" = "ingest" ]]; then
        "${INGEST_CMD[@]}"
    elif [[ "$INGEST" = "search" ]]; then
        # BEAM uses beam_evaluate.py for rubric-based evaluation
        # Other benchmarks use evaluate.py for standard LLM judge evaluation
        if [ "$TEST" != "beam" ]; then
            EVALUATE_CMD=("${PYTHON_CMD[@]}" "$SCRIPT_DIR/evaluate.py" --data-path "$RESULT_FILE" --target-path "$EVAL_FILE" --config-path "$CONFIG_FILE")
            if [ -n "${JUDGE_CONCURRENCY:-}" ]; then
                EVALUATE_CMD+=(--max_workers "$JUDGE_CONCURRENCY")
            fi
        fi
        if ! check_python_modules pandas; then
            echo "generate_scores.py requires pandas for final score generation."
            exit 1
        fi
        "${SEARCH_CMD[@]}"
        "${EVALUATE_CMD[@]}"
        "${PYTHON_CMD[@]}" "$SCRIPT_DIR/generate_scores.py" --data-path "$EVAL_FILE" > "$FINAL_SCORE_FILE"
        cat "$FINAL_SCORE_FILE"
    else
        echo "Unknown RUN_TYPE: $INGEST"
        show_help "$TEST"
    fi
}

if [ "$#" -lt 1 ]; then
    echo "Error: missing TEST argument"
    show_help all
fi

parse_optional_flags "$@"
set -- "${POSITIONAL_ARGS[@]}"

TEST="${1:-}"

# Global help
if [[ "$TEST" == "-h" || "$TEST" == "--help" ]]; then
    show_help all
fi

# Test-specific help
if [[ "${2:-}" == "-h" || "${2:-}" == "--help" ]]; then
    show_help "$TEST"
fi

validate_args "$@"

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
shopt -s nocasematch

run_test "$@"

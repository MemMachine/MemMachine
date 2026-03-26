#!/usr/bin/env bash

usage_locomo() {
    echo "Locomo Usage: $0 locomo RESULT_POSTFIX RUN_TYPE TEST_TARGET [options]"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [retrieval_skill | llm]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Max concurrent search requests (default: 1)"
    echo "  --judge-concurrency N"
    echo "                     Max concurrent LLM judge workers (default: 30)"
    echo "  --config PATH      Path to benchmark_config.yml for answer/evaluation model"
    exit 1
}

usage_wiki() {
    echo "WikiMultihop Usage: wikimultihop $0 RESULT_POSTFIX RUN_TYPE TEST_TARGET LENGTH [options]"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [retrieval_skill | llm]"
    echo "  LENGTH            Number of examples to run [1 - 12576]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Max concurrent search requests (default: 10)"
    echo "  --judge-concurrency N"
    echo "                     Max concurrent LLM judge workers (default: 30)"
    echo "  --config PATH      Path to benchmark_config.yml for answer/evaluation model"
    exit 1
}

usage_hotpotqa() {
    echo "HotpotQA Usage: $0 hotpotqa RESULT_POSTFIX RUN_TYPE SPLIT_NAME TEST_TARGET LENGTH [options]"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  SPLIT_NAME        Dataset split name [train | validation]. Train set contains 19.9%"
    echo "                      easy, 62.8% medium, 17.3% hard questions. Validation set contains"
    echo "                      hard questions only."
    echo "  TEST_TARGET       [retrieval_skill | llm]"
    echo "  LENGTH            Number of examples to run [train set 1 - 90447 | validation set 1 - 7405]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Max concurrent search requests (default: 30)"
    echo "  --judge-concurrency N"
    echo "                     Max concurrent LLM judge workers (default: 30)"
    echo "  --config PATH      Path to benchmark_config.yml for answer/evaluation model"
    exit 1
}

usage_longmemeval() {
    echo "LongMemEval Usage: $0 longmemeval RESULT_POSTFIX RUN_TYPE SPLIT_NAME TEST_TARGET LENGTH [options]"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  SPLIT_NAME        Dataset split name, e.g. longmemeval_s_cleaned"
    echo "  TEST_TARGET       [retrieval_skill | llm]"
    echo "  LENGTH            Number of examples to run [1 - split size]"
    echo "Options:"
    echo "  --search-concurrency N"
    echo "                     Max concurrent search requests (default: 30)"
    echo "  --judge-concurrency N"
    echo "                     Max concurrent LLM judge workers (default: 30)"
    echo "  --config PATH      Path to benchmark_config.yml for answer/evaluation model"
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
        ""|all)
            echo "Usage: $0 TEST [args...]"
            echo
            echo "Available TEST values:"
            echo "  locomo"
            echo "  wikimultihop"
            echo "  hotpotqa"
            echo "  longmemeval"
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
SEARCH_CONCURRENCY=""
JUDGE_CONCURRENCY=""
BENCHMARK_CONFIG=""

parse_optional_flags() {
    POSITIONAL_ARGS=()
    SEARCH_CONCURRENCY=""
    JUDGE_CONCURRENCY=""
    BENCHMARK_CONFIG=""

    while [ "$#" -gt 0 ]; do
        case "$1" in
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
            --config)
                if [ "$#" -lt 2 ]; then
                    echo "Error: --config requires a path"
                    exit 1
                fi
                BENCHMARK_CONFIG="$2"
                shift 2
                ;;
            --config=*)
                BENCHMARK_CONFIG="${1#*=}"
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
            ;;
        wikimultihop)
            if [ "$#" -ne 5 ]; then
                show_help wikimultihop
            fi
            ;;
        hotpotqa)
            if [ "$#" -ne 6 ]; then
                show_help hotpotqa
            fi
            ;;
        longmemeval)
            if [ "$#" -ne 6 ]; then
                show_help longmemeval
            fi
            ;;
        *)
            echo "Unknown test: $TEST"
            show_help all
            ;;
    esac
}

run_test() {
    TEST="$1"
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
        *)
            echo "Unknown test: $TEST"
            show_help all
            ;;
    esac

    SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/common/src:${REPO_ROOT}/packages/server/src:${REPO_ROOT}/packages/client/src${PYTHONPATH:+:${PYTHONPATH}}"
    mkdir -p ${SCRIPT_DIR}/result/final_score
    RESULT_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_output_${RESULT_POSTFIX}.json"
    EVAL_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_evaluation_metrics_${RESULT_POSTFIX}.json"
    FINAL_SCORE_FILE="${SCRIPT_DIR}/result/final_score/${TEST}_${TEST_TARGET}_${RESULT_POSTFIX}.result"
    SESSION_ID="${TEST}_${RESULT_POSTFIX}"

    rm -f "$RESULT_FILE" "$EVAL_FILE" "$FINAL_SCORE_FILE" 

    CONCURRENCY_ARGS=()
    if [ -n "${SEARCH_CONCURRENCY:-}" ]; then
        CONCURRENCY_ARGS+=(--concurrency "$SEARCH_CONCURRENCY")
    fi

    JUDGE_ARGS=()
    if [ -n "${JUDGE_CONCURRENCY:-}" ]; then
        JUDGE_ARGS+=(--max_workers "$JUDGE_CONCURRENCY")
    fi

    CONFIG_ARGS=()
    if [ -n "${BENCHMARK_CONFIG:-}" ]; then
        CONFIG_ARGS+=(--config "$BENCHMARK_CONFIG")
    fi

    case "$TEST" in
        locomo)
            INGEST_CMD=(uv run python -u "$SCRIPT_DIR/locomo_ingest.py" --data-path "$SCRIPT_DIR/../data/locomo10.json")
            SEARCH_CMD=(uv run python -u "$SCRIPT_DIR/locomo_search.py" --data-path "$SCRIPT_DIR/../data/locomo10.json" --eval-result-path "$RESULT_FILE" --test-target "$TEST_TARGET" "${CONCURRENCY_ARGS[@]}" "${CONFIG_ARGS[@]}")
            ;;
        wikimultihop)
            INGEST_CMD=(uv run python -u "$SCRIPT_DIR/wikimultihop_ingest.py" --data-path "$SCRIPT_DIR/../data/wikimultihop.json" --length "$LENGTH" --session-id "$SESSION_ID")
            SEARCH_CMD=(uv run python -u "$SCRIPT_DIR/wikimultihop_search.py" --data-path "$SCRIPT_DIR/../data/wikimultihop.json" --eval-result-path "$RESULT_FILE" --test-target "$TEST_TARGET" --length "$LENGTH" --session-id "$SESSION_ID" "${CONCURRENCY_ARGS[@]}" "${CONFIG_ARGS[@]}")
            ;;
        hotpotqa)
            INGEST_CMD=(uv run python -u "$SCRIPT_DIR/hotpotQA_test.py" --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID")
            SEARCH_CMD=(uv run python -u "$SCRIPT_DIR/hotpotQA_test.py" --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID" "${CONCURRENCY_ARGS[@]}" "${CONFIG_ARGS[@]}")
            ;;
        longmemeval)
            INGEST_CMD=(uv run python -u "$SCRIPT_DIR/longmemeval_test.py" --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID")
            SEARCH_CMD=(uv run python -u "$SCRIPT_DIR/longmemeval_test.py" --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID" "${CONCURRENCY_ARGS[@]}" "${CONFIG_ARGS[@]}")
            ;;
    esac

    if [[ "$INGEST" = "ingest" ]]; then
        "${INGEST_CMD[@]}"
    elif [[ "$INGEST" = "search" ]]; then
        "${SEARCH_CMD[@]}"
        uv run python "$SCRIPT_DIR/evaluate.py" --data-path "$RESULT_FILE" --target-path "$EVAL_FILE" "${JUDGE_ARGS[@]}" "${CONFIG_ARGS[@]}"
        uv run python "$SCRIPT_DIR/generate_scores.py" --data-path "$EVAL_FILE" > "$FINAL_SCORE_FILE"
        cat "$FINAL_SCORE_FILE"
    else
        echo "Unknown RUN_TYPE: $INGEST"
        show_help "$TEST"
    fi
}

parse_optional_flags "$@"
set -- "${POSITIONAL_ARGS[@]}"

if [ "$#" -lt 1 ]; then
    echo "Error: missing TEST argument"
    show_help all
fi

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

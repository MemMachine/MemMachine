#!/usr/bin/env bash

usage_locomo() {
    echo "Locomo Usage: $0 locomo RESULT_POSTFIX RUN_TYPE TEST_TARGET"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [memmachine | retrieval_agent | llm]"
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
    mkdir -p ${SCRIPT_DIR}/result/final_score
    RESULT_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_output_${RESULT_POSTFIX}.json"
    EVAL_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_evaluation_metrics_${RESULT_POSTFIX}.json"
    FINAL_SCORE_FILE="${SCRIPT_DIR}/result/final_score/${TEST}_${TEST_TARGET}_${RESULT_POSTFIX}.result"
    SESSION_ID="${TEST}_${RESULT_POSTFIX}"

    rm -f "$RESULT_FILE" "$EVAL_FILE" "$FINAL_SCORE_FILE"

    case "$TEST" in
        locomo)
            INGEST_CMD=(python -u "$SCRIPT_DIR/locomo_ingest.py" --data-path "$SCRIPT_DIR/../data/locomo10.json" --config-path "$CONFIG_FILE")
            SEARCH_CMD=(python -u "$SCRIPT_DIR/locomo_search.py" --data-path "$SCRIPT_DIR/../data/locomo10.json" --eval-result-path "$RESULT_FILE" --test-target "$TEST_TARGET" --config-path "$CONFIG_FILE")
            ;;
        wikimultihop)
            INGEST_CMD=(python -u "$SCRIPT_DIR/wikimultihop_ingest.py" --data-path "$SCRIPT_DIR/../data/wikimultihop.json" --length "$LENGTH" --config-path "$CONFIG_FILE")
            SEARCH_CMD=(python -u "$SCRIPT_DIR/wikimultihop_search.py" --data-path "$SCRIPT_DIR/../data/wikimultihop.json" --eval-result-path "$RESULT_FILE" --test-target "$TEST_TARGET" --length "$LENGTH" --config-path "$CONFIG_FILE")
            ;;
        hotpotqa)
            INGEST_CMD=(python -u "$SCRIPT_DIR/hotpotQA_test.py" --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --config-path "$CONFIG_FILE")
            SEARCH_CMD=(python -u "$SCRIPT_DIR/hotpotQA_test.py" --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --config-path "$CONFIG_FILE")
            ;;
        longmemeval)
            INGEST_CMD=(uv run python -u "$SCRIPT_DIR/longmemeval_test.py" --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID" --config-path "$CONFIG_FILE")
            SEARCH_CMD=(uv run python -u "$SCRIPT_DIR/longmemeval_test.py" --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET" --session-id "$SESSION_ID" --config-path "$CONFIG_FILE")
            ;;
    esac

    if [[ "$INGEST" = "ingest" ]]; then
        "${INGEST_CMD[@]}"
    elif [[ "$INGEST" = "search" ]]; then
        "${SEARCH_CMD[@]}"
        python "$SCRIPT_DIR/evaluate.py" --data-path "$RESULT_FILE" --target-path "$EVAL_FILE" --config-path "$CONFIG_FILE"
        python "$SCRIPT_DIR/generate_scores.py" --data-path "$EVAL_FILE" > "$FINAL_SCORE_FILE"
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

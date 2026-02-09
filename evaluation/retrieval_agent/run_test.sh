#!/bin/bash

usage_locomo() {
    echo "Locomo Usage: $0 locomo RESULT_POSTFIX RUN_TYPE TEST_TARGET"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [memmachine | retrieval_agent]"
    exit 1
}

usage_wiki() {
    echo "WikiMultihop Usage: wikimultihop $0 RESULT_POSTFIX RUN_TYPE TEST_TARGET LENGTH"
    echo
    echo "Arguments:"
    echo "  RESULT_POSTFIX    Custom postfix for output files"
    echo "  RUN_TYPE          Run ingestion or search [ingest | search]"
    echo "  TEST_TARGET       [memmachine | retrieval_agent]"
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
    echo "  TEST_TARGET       [memmachine | retrieval_agent]"
    echo "  LENGTH            Number of examples to run [train set 1 - 90447 | validation set 1 - 7405]"
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
        ""|all)
            echo "Usage: $0 TEST [args...]"
            echo
            echo "Available TEST values:"
            echo "  locomo"
            echo "  wikimultihop"
            echo "  hotpotqa"
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
        *)
            echo "Unknown test: $TEST"
            show_help all
            ;;
    esac

    SCRIPT_DIR=$(dirname "$0")
    mkdir -p ${SCRIPT_DIR}/result/final_score
    RESULT_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_output_${RESULT_POSTFIX}.json"
    EVAL_FILE="${SCRIPT_DIR}/result/${TEST}_${TEST_TARGET}_evaluation_metrics_${RESULT_POSTFIX}.json"
    FINAL_SCORE_FILE="${SCRIPT_DIR}/result/final_score/${TEST}_${TEST_TARGET}_${RESULT_POSTFIX}.result"

    rm -f "$RESULT_FILE" "$EVAL_FILE" "$FINAL_SCORE_FILE" 

    case "$TEST" in
        locomo)
            INGEST_CMD=(python -u locomo_ingest.py --data-path $SCRIPT_DIR/../data/locomo10.json)
            SEARCH_CMD=(python -u locomo_search.py --data-path $SCRIPT_DIR/../data/locomo10.json --target-path "$RESULT_FILE" --test-target "$TEST_TARGET")
            ;;
        wikimultihop)
            INGEST_CMD=(python -u wikimultihop_ingest.py --data-path $SCRIPT_DIR/../data/wikimultihop.json --length "$LENGTH")
            SEARCH_CMD=(python -u wikimultihop_search.py --data-path $SCRIPT_DIR/../data/wikimultihop.json --target-path "$RESULT_FILE" --test-target "$TEST_TARGET" --length "$LENGTH")
            ;;
        hotpotqa)
            INGEST_CMD=(python -u hotpotQA_test.py --run-type ingest --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET")
            SEARCH_CMD=(python -u hotpotQA_test.py --run-type search --eval-result-path "$RESULT_FILE" --length "$LENGTH" --split-name "$SPLIT_NAME" --test-target "$TEST_TARGET")
            ;;
    esac

    if [[ "$INGEST" = "ingest" ]]; then
        "${INGEST_CMD[@]}"
    elif [[ "$INGEST" = "search" ]]; then
        "${SEARCH_CMD[@]}"
        python evaluate.py --data-path "$RESULT_FILE" --target-path "$EVAL_FILE"
        python generate_scores.py --data-path "$EVAL_FILE" > "$FINAL_SCORE_FILE"
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

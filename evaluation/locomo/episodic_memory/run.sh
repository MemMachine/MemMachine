TEST_NAME=$1
RESULT_FILE="./result/output_${TEST_NAME}.json"
EVAL_FILE="./result/evaluation_metrics_${TEST_NAME}.json"
FINAL_SCORE_FILE="./result/${TEST_NAME}.result"

rm -f $RESULT_FILE $EVAL_FILE $FINAL_SCORE_FILE

set -xe
export OPENAI_API_KEY=key
export PYTHONUNBUFFERED=1
#python3 -u locomo_ingest.py --data-path ../locomo10.json | tee ingest.log
python -u locomo_search.py --data-path ../locomo10.json --target-path $RESULT_FILE
python locomo_evaluate.py --input_file $RESULT_FILE --output_file $EVAL_FILE
python generate_scores.py $EVAL_FILE > $FINAL_SCORE_FILE
cat $FINAL_SCORE_FILE

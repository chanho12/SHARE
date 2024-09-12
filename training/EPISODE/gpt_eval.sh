CUDA_VISIBLE_DEVICES=1 #0, 1

API_KEY=
ACCUMULATE_PATH=
UPDATE_PATH=
MODEL_NAME="gpt-4o" # gpt-3.5_turbo, gpt_4o
RESULT='update_respectly_v1.json'   
CRITERION=''

python gpt_eval.py  --api_key $API_KEY --model_name $MODEL_NAME  --accumulate $ACCUMULATE_PATH --update $UPDATE_PATH --result $RESULT --criterion $CRITERION
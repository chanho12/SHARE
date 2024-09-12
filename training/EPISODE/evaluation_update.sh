export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_llama6_final_2_session.json'  #evaluation dataset path
MODEL_NAME='llama'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME

export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_llama6_final_3_session.json'  #evaluation dataset path
MODEL_NAME='llama'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME


export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_llama6_final_4_session.json'  #evaluation dataset path
MODEL_NAME='llama'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME


export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_llama6_final_5_session.json'  #evaluation dataset path
MODEL_NAME='llama'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME


export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_gemma6_final_2_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME

export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_gemma6_final_3_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME


export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_gemma6_final_4_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME


export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/EPISODE-update/results/new_update_v5_num6_gemma6_final_5_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME

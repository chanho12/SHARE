export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/training/EPISODE-update/results/update_num6_llama_2_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update_copy.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME


export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/training/EPISODE-update/results/update_num6_llama_3_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update_copy.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME
export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/training/EPISODE-update/results/update_num6_llama_4_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update_copy.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME
export CUDA_VISIBLE_DEVICES=0 # 0,1 

DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/training/EPISODE-update/results/update_num6_llama_5_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_update_copy.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME

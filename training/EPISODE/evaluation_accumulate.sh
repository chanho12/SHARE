export CUDA_VISIBLE_DEVICES=1 # 0,1 

DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/training/EPISODE/gemma_dataset/update_gemma__2_session.json'  #evaluation dataset path
MODEL_NAME='gemma'
TASK_NAME='w_tag'

python evaluation_accumulate.py --data_path $DATA_PATH --model_name $MODEL_NAME --task_name $TASK_NAME
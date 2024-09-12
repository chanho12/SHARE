export CUDA_VISIBLE_DEVICES=0 #0, 1


DATA_PATH="/home/chanho/Model/SHARE/Refactorizing/training/UPDATE/dataset_v4/test_data.json"    
DATA_NAME="inference_update_v1.json" 
MODEL_PATH="chano12/update_llama_v3"   
TASK_NAME='persona' #persona, personal, shared
python inference_update.py --data_path $DATA_PATH --data_name $DATA_NAME --model_path $MODEL_PATH --task_name $TASK_NAME
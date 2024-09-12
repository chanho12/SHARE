export CUDA_VISIBLE_DEVICES=1 # 0,1 

TASK_NAME='baseline' #wo_tag, w_tag, baseline
MODEL_NAME='base_llama' #llama, gemma, base_llama, base_gemma
DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/result/dataset/test_without_tag.json'  #evaluation dataset path
MODEL_TAG='model_tag'

python evaluation_automatic.py --task_name $TASK_NAME --model_name $MODEL_NAME --data_path $DATA_PATH --model_tag $MODEL_TAG

export CUDA_VISIBLE_DEVICES=1 # 0,1 

TASK_NAME='wo_tag' #wo_tag, w_tag, baseline
MODEL_NAME='llama' #llama, gemma, base_llama, base_gemma
DATA_PATH='/home/chanho/Model/SHARE/Refactorizing/result/dataset/test_without_tag.json'  #evaluation dataset path
MODEL_TAG='model_tag'

python evaluation_automatic.py --task_name $TASK_NAME --model_name $MODEL_NAME --data_path $DATA_PATH --model_tag $MODEL_TAG

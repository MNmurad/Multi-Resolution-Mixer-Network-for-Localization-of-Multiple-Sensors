#!/bin/bash
# Create WPMixer logs directory if it doesn't exist
if [ ! -d "./logs/BridgeModel" ]; then
    mkdir ./logs/BridgeModel
fi

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Datasets and prediction lengths
model_name=BridgeModel
data_path=test-1.csv
seq_lens=256
pred_lens=1
learning_rates=0.00982476146087077
batches=32
tfactors=7
dfactors=7
epochs=60
dropouts=0.2
embedding_dropouts=0.1
patch_lens=16
strides=8
lradjs=type3
d_models=16
patiences=12
wavelets=sym3
weight_decay=0.05
levels=1

# Training
log_file="logs/${model_name}/full_hyperSearch_result_${data_path}_${pred_lens}.log"
python -u main_Mixer.py \
	--model $model_name \
	--train_cf_optimization 0 \
	--test_cf_optimization 0 \
	--task_name long_term_forecast \
	--data_path $data_path \
	--seq_len $seq_lens \
	--pred_len $pred_lens \
	--loss mse \
	--d_model $d_models \
	--tfactor $tfactors \
	--dfactor $dfactors \
	--wavelet $wavelets \
	--level $levels \
	--patch_len $patch_lens \
	--stride $strides \
	--batch_size $batches \
	--learning_rate $learning_rates \
	--lradj $lradjs \
	--dropout $dropouts \
	--embedding_dropout $embedding_dropouts \
	--weight_decay $weight_decay \
	--patience $patiences \
	--train_epochs $epochs > $log_file
	
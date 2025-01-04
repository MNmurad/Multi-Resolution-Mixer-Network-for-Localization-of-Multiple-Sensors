if [ ! -d "./logs/DataGeneration" ]; then
    mkdir ./logs/DataGeneration
fi


export CUDA_VISIBLE_DEVICES=0
# General

# train with seed 42
vel_model=train
seed=42
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
# train with seed 52
vel_model=train
seed=52
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
# train with seed 62
vel_model=train
seed=62
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
		
# Validation
vel_model=vali
seed=42
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
		
# test-1
vel_model=test-1
seed=42
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log

# test-2
vel_model=test-2
seed=42
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
		
# test-3
vel_model=test-3
seed=42
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
		
# test-4
vel_model=test-4
seed=42
python -u main_dataGeneration_and_UKF.py \
		--seed $seed \
		--task moving_dataset \
		--velocity_model_id $vel_model \
		--initial_nodes_position -8.5 10.0 -7.0 10.0 -4.5 10.0 -1.5 10.0 1.0 9.5 2.5 9.3 5.0 9.8 6.5 9.8 8.5 9.4 \
		--interrogators_position -12 12 10 1.571 1.571 1 12 12 10 1.571 1.571 1 -12 -12 10 1.571 1.571 1 12 -12 10 1.571 1.571 1 0 0 10 1.571 1.571 1 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/DataGeneration/${vel_model}_${seed}.log
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


export CUDA_VISIBLE_DEVICES=0
# General

seed=42
task=ukf
python -u main.py \
		--seed $seed \
		--task $task \
		--ukf_alpha 1.0 \
		--ukf_beta 2.0 \
		--ukf_kappa 0.01 \
		--ukf_R_factor 1e-8 \
		--ukf_P_factor 1e-8 \
		--ukf_Q_factor 1e-2 \
		--x_boundary -10 10 \
		--y_boundary -10 10 \
		--slope_angle 30 \
		--dt 0.005 >logs/${task}_${vel_model}_${seed}.log


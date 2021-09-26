
### Section 1 (Behavior Cloning)
Ant
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_d
ata cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --num_agent_train_steps_per_iter 2000
```
Hopper
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 -
-expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --ep_len 1000 --num_agent_train_steps_per_iter 2000
```

### Section 2 (DAgger)
Ant
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_
dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --ep_len 1000
```
Hopper
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name dagger_ant --n_iter 10
 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --ep_len 1000
```

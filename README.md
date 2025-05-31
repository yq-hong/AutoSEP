# Unlabeled Data Improves Fine-Grained Image Zero-shot Classification with Multimodal LLMs

## Important environment requirements
```
conda create -n autosep python=3.9
pip install -r requirements.txt
```

## Models
For proprietary models, set your API key in `config.py`.  
To use open-source models like **Qwen2-VL-72B-Instruct**, we use the **SGLang** framework.  
Serve the model by running:  
```
bash scripts/launch/Qwen2-VL-72B-Instruct.sh
```

## AutoSEP (Automatic Self-Enhancing Prompt Learning)
### AutoSEP optimization
```
cd autosep
python main.py --data_dir ${data_dir} --model gemini --gradient_mode gemini --task_name ${task_name} --n_train 30 --test_eval --rounds 6 --beam_size 4 --minibatch_size 50 --n_gradients 4 --mc_samples_per_step 1 --max_expansion_factor 5 --out_num 1
```

### Prompt evaluation
Instance-level classification for prompts during the AutoSEP optimization:

(Setting `--parallel` will speed up the process.)
```
cd autosep
python llm_text_compare.py --evaluate --generate --parallel --result_folder autosep --data_dir ${data_dir} --model gemini --exp 1 --task_name ${task_name} --mode train --n_test 30 --n_compare 10
```

Clase-wise classification:
```
python classification.py --attributes --generate --parallel --result_folder autosep --data_dir ${data_dir} --model gemini --exp 1 --task_name ${task_name} --mode test --n_test 30 --prompt_idx 10 --out_num 1
```

## Baselines
### Optimization-free
#### Vanilla zero-shot
```
python multi_zero_shot.py --parallel --data_dir ${data_dir} --model gemini --task_name ${task_name} --mode test --n_test 30 --out_num 1
```

#### Zero-shot with descriptions
```
python multi_zero_shot.py --attributes --generate --parallel --data_dir ${data_dir} --model gemini --task_name ${task_name} --mode test --n_test 30 --out_num 1
```

#### Zero-shot with majority vote
```
cd baseline
python majority_vote.py --parallel --data_dir ${data_dir} --model gemini --task_name ${task_name} --mode test --n_test 30 --temperature 0.7 --n_votes 5 --out_num 1
```

#### Few-shot with random labels
```
cd baseline
python multi_random_label.py --random_labels --parallel --data_dir ${data_dir} --model gemini --task_name ${task_name} --mode test --n_test 30 --n_examples 5 --seed 1
```

#### Multiple images display
```
cd baseline
python multi_random_label.py --parallel --data_dir ${data_dir} --model gemini --task_name ${task_name} --mode test --n_test 30 --n_examples 5 --seed 1
```

#### K-means clustering
```
cd baseline
python cluster_img.py --data_dir ${data_dir} --model gemini --task_name ${task_name} --mode test --n_test 30 --device cuda:0 --n_clusters 7 --n_examples 3 --seed 1
```

### Optimization-based
#### Optimization with random labels
```
cd baseline
python main.py --model gemini --gradient_mode gemini --task_name ${task_name} --data_dir ${data_dir} --n_train 30 --test_eval --method random_label --rounds 6 --beam_size 4 --minibatch_size 50 --out_num 1
```

#### Optimization with majority vote
```
cd baseline
python main.py --model gemini --gradient_mode gemini --task_name ${task_name} --data_dir ${data_dir} --n_train 30 --test_eval --method majority_vote --temperature 0.7 --n_votes 5 --rounds 6 --beam_size 4 --minibatch_size 50 --out_num 2
```

#### SPO
```
cd spo
python main.py --model gemini --gradient_mode gemini --task_name ${task_name} --data_dir ${data_dir} --n_train 30 --test_eval --rounds 10 --beam_size 1 --minibatch_size 7 --n_gradients 1 --errors_per_gradient 3 --mc_samples_per_step 0 --max_expansion_factor 1 --out_num 1
```

This repo is originally based on https://github.com/microsoft/LMOps/tree/main/prompt_optimization
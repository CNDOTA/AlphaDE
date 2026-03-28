# AlphaDE

AlphaDE: boosting in silico directed evolution with fine-tuned protein language model and tree search.

AlphaDE combines Monte Carlo Tree Search (MCTS) with AlphaZero-style architecture and fine-tuned protein language models (ESM2) with homology sequences to evolve proteins toward higher fitness.

## Table of Contents

1. [Installation](#1-installation)
2. [Model Setup](#2-model-setup)
3. [Fine-tuning](#3-fine-tuning)
4. [MCTS Inference](#4-mcts-inference)
5. [Project Structure](#5-project-structure)

## 1. Installation

### 1.1 Conda Environment

```bash
conda create -n alphade python=3.10
conda activate alphade
pip install -r requirements.txt
pip install transformers[torch]==4.48.3
pip install fair-esm
pip install transformers[sentencepiece]
pip install deepblast
conda install -c conda-forge -c bioconda hhsuite
pip install numba
pip install tape_proteins
pip install sequence-models
```

## 2. Model Setup

### 2.1 Download TAPE Oracle

1. Download oracle weights from: https://drive.usercontent.google.com/download?id=1uy9zgtJ60Z83LCbm7Z_AoksAkmK4CcsC

2. Unzip and place the `tape_landscape` folder under the `AlphaDE` directory.

### 2.2 Download ESM2 Model

1. Download ESM2-35M model from: https://huggingface.co/facebook/esm2_t12_35M_UR50D

2. Place the `esm2_t12_35M_UR50D` folder under the `AlphaDE` directory.

## 3. Fine-tuning

### 3.1 Prepare Dataset

The homology dataset for fine-tuning is located at:
```
AlphaDE/data/$task/percent_data/$task_sequences_bottom_percent_1.0.txt
```

### 3.2 Run Fine-tuning

```bash
cd ./finetuning
python -u run_mlm.py \
    --model_name_or_path $pretrained_model_path \
    --train_file $finetuning_data_path \
    --validation_file $finetuning_data_path \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_strategy epoch \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir $finetuned_model_path > log.txt
```

## 4. MCTS Inference

### 4.1 Configure Model Paths

Edit `vocab.py` to set the estimated maximum fitness (for scaling the reward) and fine-tuned model path:

```python
MAX_TAPE = $predefined_task_maximum_fitness
ESM2_PATH = $finetuned_model_path
```

### 4.2 Run AlphaDE

Results are saved in the `AlphaDE/results` folder.

| Task  | Command                                                                 |
|-------|-------------------------------------------------------------------------|
| avGFP | `python train.py --task=avGFP --gpus=0 --idx=1 > avGFP_ft_35M_1.log 2>&1 &` |
| AAV   | `python train.py --task=AAV --gpus=0 --idx=1 > AAV_ft_35M_1.log 2>&1 &`   |
| TEM   | `python train.py --task=TEM --gpus=0 --idx=1 > TEM_ft_35M_1.log 2>&1 &`   |
| E4B   | `python train.py --task=E4B --gpus=0 --idx=1 > E4B_ft_35M_1.log 2>&1 &`    |
| AMIE  | `python train.py --task=AMIE --gpus=0 --idx=1 > AMIE_ft_35M_1.log 2>&1 &`  |
| LGK   | `python train.py --task=LGK --gpus=0 --idx=1 > LGK_ft_35M_1.log 2>&1 &`    |
| PAB1  | `python train.py --task=PAB1 --gpus=0 --idx=1 > PAB1_ft_35M_1.log 2>&1 &`  |
| UBE2I | `python train.py --task=UBE2I --gpus=0 --idx=1 > UBE2I_ft_35M_1.log 2>&1 &` |

### 4.3 Optional Parameters

| Parameter        | Default | Description                          |
|------------------|---------|--------------------------------------|
| `--cpuct`        | 10      | MCTS exploration constant            |
| `--tree_depth`   | 100     | Maximum mutation search depth        |
| `--rollout_number` | 200   | Number of MCTS playouts             |

## 5. Project Structure

```
AlphaDE/
├── train.py                         # Main training pipeline
├── vocab.py                         # Configuration (alphabet, scores, paths)
├── plm_v_net.py                    # Policy-Value Network
├── mcts_alphaZero_mutate_expand.py  # MCTS AlphaZero mutator
├── sequence_env_m_p.py              # Sequence environment
├── esm1b_landscape.py              # ESM1b landscape model
├── finetuning/
│   └── run_mlm.py                   # MLM fine-tuning script
├── data/                            # Fine-tuning datasets
├── tape_landscape/                  # TAPE oracle models
├── esm2_t12_35M_UR50D/              # ESM2 pretrained model
└── results/                        # Output results
```

## Citation

If you use AlphaDE in your research, please cite the original paper.

```bibtex
@article{yang2025alphade,
  title={AlphaDE: boosting in silico directed evolution with fine-tuned protein language model and tree search},
  author={Yang, Yaodong and Wang, Yang and Li, Jinpeng and Guo, Pei and Han, Da and Chen, Guangyong and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2511.09900},
  year={2025}
}
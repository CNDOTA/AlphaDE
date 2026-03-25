# 1. Install

## 1.1 Conda environment
Using the following script to create a conda environment
```bash
conda create -n alphade python=3.10
conda activate alphade

pip install -r requirements.txt

pip install transformers[torch]
pip install fair-esm
pip install transformers[sentencepiece]
pip install deepblast

conda install -c conda-forge -c bioconda hhsuite

pip install numba
pip install tape_proteins
```

## 1.2 Download TAPE oracle and pretrained ESM2 models
oracle download: https://github.com/HeliXonProtein/proximal-exploration/blob/main/download_landscape.sh

ESM2-35M model download: https://huggingface.co/facebook/esm2_t12_35M_UR50D

# 2. Finetune
## 2.1 Prepare the fine-tuning dataset
The homology dataset for fine-tuning is available at 
"AlphaDE/data/\$task/percent_data/\$task_sequences_bottom_percent_1.0.txt"

## 2.2. Fine-tuning protein language models
In the folder of "AlphaDE/finetuning", using the following script as
```commandline
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

# 3. MCTS Inference
## 3.1. Model path configuration
We need to configure the estimated maximum fitness (for scaling the reward) and fine-tuned model path in "vocab.py" first.
```commandline
MAX_TAPE = $predefined_task_maximum_fitness
ESM2_PATH = $finetuned_model_path
```

## 3.2. Run AlphaDE
### Task avGFP
```bash
python train.py --task=avGFP --gpus=0 --idx=1 > avGFP_ft_35M_1.log 2>&1 &
```
### Task AAV
```bash
python train.py --task=AAV --gpus=0 --idx=1 > AAV_ft_35M_1.log 2>&1 &
```
### Task TEM
```bash
python train.py --task=TEM --gpus=0 --idx=1 > TEM_ft_35M_1.log 2>&1 &
```
### Task E4B
```bash
python train.py --task=E4B --gpus=0 --idx=1 > E4B_ft_35M_1.log 2>&1 &
```
### Task AMIE
```bash
python train.py --task=AMIE --gpus=0 --idx=1 > AMIE_ft_35M_1.log 2>&1 &
```
### Task LGK
```bash
python train.py --task=LGK --gpus=0 --idx=1 > LGK_ft_35M_1.log 2>&1 &
```
### Task PAB1
```bash
python train.py --task=PAB1 --gpus=0 --idx=1 > PAB1_ft_35M_1.log 2>&1 &
```
### Task UBE2I
```bash
python train.py --task=UBE2I --gpus=0 --idx=1 > UBE2I_ft_35M_1.log 2>&1 &
```
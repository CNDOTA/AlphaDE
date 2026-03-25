AAS = "ILVAGMFYWEDQNHCRKSTP"
SCORE_LIST = ['TAPE']
SCORE_DIM = len(SCORE_LIST)

# for avGFP
MAX_TAPE = 4.0
ESM2_PATH = "./finetuning/esm2_train_output/avgfp/esm2_t12_35M_UR50D_per1.0/checkpoint-6753"  # finetune 3 epochs

# for AAV
# MAX_TAPE = 10.0
# ESM2_PATH = "./finetuning/esm2_train_output/aav/esm2_t12_35M_UR50D_per1.0/checkpoint-36663"  # finetune 3 epochs

# for TEM
# MAX_TAPE = 2.0
# ESM2_PATH = "./finetuning/esm2_train_output/tem/esm2_t12_35M_UR50D_per1.0/checkpoint-651"  # finetune 3 epochs

# for E4B
# MAX_TAPE = 10.0
# ESM2_PATH = "./finetuning/esm2_train_output/e4b/esm2_t12_35M_UR50D_per1.0/checkpoint-12288"  # finetune 3 epochs

# for AMIE
# MAX_TAPE = 1.0
# ESM2_PATH = "./finetuning/esm2_train_output/amie/esm2_t12_35M_UR50D_per1.0/checkpoint-804"  # finetune 3 epochs

# for LGK
# MAX_TAPE = 1.0
# ESM2_PATH = "./finetuning/esm2_train_output/lgk/esm2_t12_35M_UR50D_per1.0/checkpoint-957"  # finetune 3 epochs

# for PAB1
# MAX_TAPE = 2.5
# ESM2_PATH = "./finetuning/esm2_train_output/pab1/esm2_t12_35M_UR50D_per1.0/checkpoint-4551"  # finetune 3 epochs

# for UBE2I
# MAX_TAPE = 5.0
# ESM2_PATH = "./finetuning/esm2_train_output/ube2i/esm2_t12_35M_UR50D_per1.0/checkpoint-378"  # finetune 3 epochs

ESM2_VOCAB = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

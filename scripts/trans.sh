#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

MODEL_DIR=$1

python src/translate.py \
  --model_file="${MODEL_DIR}/bleu_final_nmt_model.pt" \
  --hparams_file="${MODEL_DIR}/bleu_final_nmt_hparams.pt" \
  --test_src_file_list data/ted_processed/bel_eng/ted-test.spm8000.bel \
  --test_trg_file_list data/ted_processed/bel_eng/ted-test.spm8000.eng \
  --test_ref_file_list data/ted_processed/bel_eng/ted-test.mtok.eng \
  --test_file_idx_list "0" \
  --cuda \
  --merge_bpe \
  --beam_size=5 \
  --poly_norm_m=1 \
  --max_len=200 \
  --out_file="${MODEL_DIR}ted-test-b5m1"

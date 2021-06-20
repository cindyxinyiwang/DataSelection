#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --mem=18g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="0"

DDIR=data/

python src/rl_main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_s0/dds_tcs_bel/" \
  --train_src_file_list "$DDIR"/ted_processed/LAN_eng/ted-train.spm8000.LAN \
  --train_trg_file_list  "$DDIR"/ted_processed/LAN_eng/ted-train.spm8000.eng \
  --dev_src_file_list  "$DDIR"/ted_processed/bel_eng/ted-dev.spm8000.bel \
  --dev_trg_file_list  "$DDIR"/ted_processed/bel_eng/ted-dev.spm8000.eng \
  --dev_ref_file_list  "$DDIR"/ted_processed/bel_eng/ted-dev.mtok.eng \
  --dev_file_idx_list  "0" \
  --src_vocab_list  "$DDIR"/ted_processed/ted-train.mtok.spm8000.src.vocab \
  --trg_vocab  "$DDIR"/ted_processed/eng/ted-train.mtok.spm8000.eng.vocab \
  --lang_file langs_tiny.txt \
  --lan_dist_file "$DDIR"ted-train-vocab.mtok.sim-ngram.graph \
  --base_lan "bel" \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=20 \
  --merge_bpe \
  --eval_bleu \
  --batcher='word' \
  --batch_size 1500 \
  --raw_batch_size 1 \
  --lr_dec 1.0 \
  --lr 0.001 \
  --lr_q 0.0001 \
  --n_train_epochs=20 \
  --dropout 0.3 \
  --max_len 300 \
  --agent_subsample_line 160 \
  --train_score_episode 100 \
  --print_every 50 \
  --init_train_score_every=5000 \
  --init_load_time=5 \
  --train_score_every=20000 \
  --record_grad_step=100 \
  --data_name="ted_processed" \
  --d_hidden 32 \
  --reward_scale=0.01 \
  --grad_dist="cosine" \
  --cuda \
  --imitate_episode 20 \
  --imitate_type "heuristic" \
  --actor_type="base" \
  --feature_type="zero_one" \
  --norm_feature \
  --add_bias=0 \
  --seed 0

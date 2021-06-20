#!/bin/bash

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

VOCAB_SIZE=8000
RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/
FAIR_SCRIPTS=scripts/
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl
FILTER=mosesdecoder/scripts/training/clean-corpus-n.perl

LANS=(
  aze
  bel
  rus
  glg
  por
  slk
  ces)

for LAN in ${LANS[@]}; do
  mkdir -p "$PROC_DDIR"/"$LAN"_eng

  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.*-eng  ; do
    src=`echo $f | sed 's/-eng$//g'`
    trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`
    #if [ ! -f "$src" ]; then
    if true; then
      echo "src=$src, trg=$trg"
      python cut-corpus.py 0 < $f > $src
      python cut-corpus.py 1 < $f > $trg
    fi
  done
  for f in "$RAW_DDIR"/"$LAN"_eng/*.orig.{eng,$LAN} ; do
    f1=${f/orig/mtok}
    #if [ ! -f "$f1" ]; then
    if true; then
      echo "tokenize $f1..."
      cat $f | perl $TOKENIZER > $f1
    fi
  done
done

## learn BPE with sentencepiece for English
TRAIN_FILES=$(for LAN in "${LANS[@]}"; do echo "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng; done | tr "\n" ",")
echo "learning BPE for eng over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
      --input=$TRAIN_FILES \
      --model_prefix="$PROC_DDIR"/spm"$VOCAB_SIZE".eng \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=1.0 \
      --model_type=unigram

# train a separate BPE model for each language, then encode the data with the corresponding BPE model
for LAN in ${LANS[@]}; do
  TRAIN_FILES="$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN"
  echo "learning BPE for ${TRAIN_FILES} ..."
  python "$SPM_TRAIN" \
        --input=$TRAIN_FILES \
        --model_prefix="$PROC_DDIR"/"$LAN"_eng/spm"$VOCAB_SIZE" \
        --vocab_size=$VOCAB_SIZE \
        --character_coverage=1.0 \
        --model_type=unigram

  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".eng.model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok.eng  \
        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".prefilter.eng \

  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/"$LAN"_eng/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-train.mtok."$LAN"  \
        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".prefilter."$LAN" \

  # filter out training data longer than 200 words
  $FILTER "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".prefilter $LAN eng "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE" 1 200

  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".eng.model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok.eng  \
        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE".eng
  done
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/"$LAN"_eng/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LAN"_eng/ted-"$split".mtok."$LAN"  \
        --outputs "$PROC_DDIR"/"$LAN"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LAN"
  done
done

for LAN in ${LANS[@]}; do
    f="$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE"."$LAN"
    echo "python src/get_vocab.py < $f > $f.vocab &"
    python src/get_vocab.py < $f > $f.vocab &
done

TRAIN_FILES=$(for LAN in "${LANS[@]}"; do echo "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE"."$LAN"; done | tr "\n" " ")
echo "learning BPE for eng over ${TRAIN_FILES}..."
cat  $TRAIN_FILES >> "$PROC_DDIR"/src.spm"$VOCAB_SIZE"
python src/get_vocab.py < "$PROC_DDIR"/src.spm"$VOCAB_SIZE" > "$PROC_DDIR"/ted-train.mtok.spm"$VOCAB_SIZE".src.vocab

TRAIN_FILES=$(for LAN in "${LANS[@]}"; do echo "$PROC_DDIR"/"$LAN"_eng/ted-train.spm"$VOCAB_SIZE".eng; done | tr "\n" " ")
echo "learning BPE for eng over ${TRAIN_FILES}..."
mkdir -p "$PROC_DDIR"/eng
cat  $TRAIN_FILES >> "$PROC_DDIR"/eng/eng.spm"$VOCAB_SIZE"

python src/get_vocab.py < "$PROC_DDIR"/eng/eng.spm"$VOCAB_SIZE" > "$PROC_DDIR"/eng/ted-train.mtok.spm"$VOCAB_SIZE".eng.vocab

python src/get_trg_src_correspondence.py

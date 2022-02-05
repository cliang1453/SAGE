#! /bin/sh
# python experiments/glue/glue_prepro.py
python prepro_std.py \
  --model roberta-large \
  --root_dir /root/data/mtdnn/canonical_data \
  --task_def experiments/glue/glue_task_def.yml \
  --do_lower_case $1

CUDA_VISIBLE_DEVICES="0" python unimol/test.py --user-dir ./unimol --valid-subset test --results-path ./test --num-workers 8 \
--ddp-backend=c10d --batch-size 8 --task tta --loss in_batch_softmax --arch tta \
--fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1\
--path checkpoint_best.pt --log-interval 100 --log-format simple \
--max-pocket-atoms 511 --tta-time 1 --tta-time-p 1\
--tta-lr 0.005 --tta-lr-p 0.0001 --test-task DUDE \
--checkpoint-path checkpoint_path --mol-path ./data/DUD-E/raw/all/ \
--target-path ./data/DUD-E/raw/all/ ./data/DUD-E/
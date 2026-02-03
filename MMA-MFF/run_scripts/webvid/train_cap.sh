CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nproc_per_node=4 \
    --master_port=34659 \
    train.py \
    --cfg-path lavis/projects/malmm/cap_webvid.yaml \
    --options \
    model.arch blip2_vicuna_instruct_first_stage \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 40 \
    model.num_frames 80 \
    run.init_lr 1e-5 \
    run.max_epoch 4 \
    run.num_beams 5 \
    run.batch_size_train 8 \
    run.batch_size_eval 8 \
    run.accum_grad_iters 2 \
    run.num_workers 0 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "[]" \
    run.report_metric True \
    run.prefix train
    # run.resume_ckpt_path

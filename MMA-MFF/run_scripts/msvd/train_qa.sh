CUDA_VISIBLE_DEVICES=0,1,2,5  torchrun --nproc_per_node=4 \
    --master_port=34651 \
    train.py \
    --cfg-path lavis/projects/malmm/qa_msvd.yaml \
    --options \
    model.arch blip2_vicuna_instruct_clip2 \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 20 \
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 5 \
    run.batch_size_train 4 \
    run.batch_size_eval 4 \
    run.accum_grad_iters 4 \
    run.num_workers 0 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['val', 'test']" \
    run.report_metric True \
    run.prefix train
    # run.resume_ckpt_path

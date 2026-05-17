export VLLM_USE_MODELSCOPE=True
export VLLM_OMNI_STORAGE_PATH="./"
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export MULTI_STREAM_MEMORY_REUSE=2
export MINDIE_SD_FA_TYPE=ascend_laser_attention

vllm serve /root/.cache/modelscope/hub/models/zhaotutu12/Wan2.2-T2V-A14B-Diffusers-bf16  \
  --omni \
  --port 8099 \
  --usp 8 \
  --use-hsdp \
  --enforce-eager \
  --log-stats \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": "False"}'\
  --enable-diffusion-pipeline-profiler \
  --vae-patch-parallel-size 8 \
  --vae-use-tiling

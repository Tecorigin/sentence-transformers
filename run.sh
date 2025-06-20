#export TORCH_SDAA_RUNTIME_AUTOFALLBACK=1 
#export TORCH_SDAA_FALLBACK_OPS=embedding,embedding_dense_backward
#export TECO_LOGGING_LEVEL=15
#export TORCH_SDAA_LOG_LEVEL=Debug 
#export TECO_ENABLE_DUMP_INFO=1 
#export TECO_DUMP_INFO_OP_NAME="embedding_backward" 
#export TECO_DUMP_INFO_DUMP_DATA_IN=1 
#export TECO_DUMP_INFO_DUMP_DATA_FILE=1
#export TECO_DUMP_INFO_CORE_DUMP=1
#python test.py

export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:1024
export OMP_NUM_THREADS=4
export TORCH_SDAA_LINEAR_HIGHPREC=1
export TORCH_SDAA_BADDBMM_HIGHPREC=1
export TORCH_SDAA_BMM_HIGHPREC=1
export TORCH_SDAA_BMM_HIGHPERF=1
export TORCH_SDAA_BLAS_TRANSPOSE=0
export TORCH_SDAA_FUSED_ATTN_MEM_LIMITED=1
export TORCH_SDAA_ALIGN_NV_DEVICE=a100
export HF_ENDPOINT=https://hf-mirror.com


deepspeed --include localhost:0,1,2,3 --master_port 29501 train_example.py



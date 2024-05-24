export LD_LIBRARY_PATH="/data/data1/conda_data/envs/paddle_gpu/lib/"
export CUDA_VISIBLE_DEVICES="9"
#paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189
paddlenlp server server_with_tb:app --workers 2 --host 0.0.0.0 --port 8189


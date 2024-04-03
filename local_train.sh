# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:./

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=projects/configs/$1.py

if [ ${gpu_num} -gt 1 ]
then
    bash /home/bo.yang5/other/Sparse4D-full/tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=work_dirs/tune/$1 \

else
    python ./tools/train.py \
        ${config}
fi

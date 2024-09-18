# cd /gaozhangyang/experiments/ProteinInvBench
# CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name E3PiFold_dist_noleakcage --use_dist 1 --use_product 0 --batch_size 32 --lr 5e-5 --epoch 15 --offline 0

cd /gaozhangyang/experiments/ProteinInvBench
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/flash/bin/python ./train/main.py --ex_name E3PiFold_dist_product_noleakcage  --use_dist 1 --use_product 1 --batch_size 32 --lr 5e-5 --epoch 15 --offline 0
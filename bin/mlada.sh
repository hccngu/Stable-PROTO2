#
#dataset=20newsgroup
#data_path="../data/20news.json"
#n_train_class=8
#n_val_class=5
#n_test_class=7

dataset=amazon
data_path="../data/amazon.json"
n_train_class=10
n_val_class=5
n_test_class=9
#
#dataset=huffpost
#data_path="../data/huffpost.json"
#n_train_class=20
#n_val_class=5
#n_test_class=16
#
#
#dataset=reuters
#data_path="../data/reuters.json"
#n_train_class=15
#n_val_class=5
#n_test_class=11
#
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --test_epochs 300 \
    --val_epochs 200 \
    --train_iter 10 \
    --test_iter 15 \
    --meta_lr 1e-5 \
    --task_lr 7e-1 \
    --Comments "amazon + weight_decay:5e-4 + dropout:0.2 + lr:1e-3, 7e-2, train_episodes:8" \
    --patience 20 \
    --seed 3 \
    --notqdm \
    --weight_decay 1e-4 \
    --dropout 0.2 \
    --loss_weight 10.0 \

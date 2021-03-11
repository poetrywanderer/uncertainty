export CUDA_VISIBLE_DEVICES=0

python run_nerf_uncertainty_kde.py --config configs/scan21.txt --index_ensembles 1 # --no_reload 
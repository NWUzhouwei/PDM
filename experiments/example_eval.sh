cd evaluation

export pred_dir="/home/sxz/下载/BDM-main/experiments/outputs/train_sofa_samplebing_pc_1_pix/2025-03-06--17-07-24/sample/pred/sofa/" # (e.g., ".../sample/pred/chair")
export gt_dir="/home/sxz/下载/BDM-main/experiments/outputs/train_sofa_samplebing_pc_1_pix/2025-03-06--17-07-24/sample/gt/sofa/" # (e.g., ".../sample/gt/chair")

echo "----------"
echo "sample_chair_pc2_r2n2_0.1"

python evaluation_cd.py \
    --pred_dir ${pred_dir} \
    --gt_dir ${gt_dir} \
    --seed 2003

python evaluation_f1.py \
    --pred_dir ${pred_dir} \
    --gt_dir ${gt_dir} \
    --seed 2003
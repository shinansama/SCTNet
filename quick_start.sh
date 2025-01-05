#
pip install "mmsegmentation==v0.26.0"
pip install yapf==0.40.1


python tools/train.py configs/sctnet/pets/sctnet-b_8x4_160k_pets.py
python tools/train.py configs/sctnet/pets/sctnet-s_8x4_160k_pets.py
python tools/train.py configs/sctnet/pets/sctnet-s_nt_8x4_160k_pets.py

bash tools/train.sh configs/sctnet/pets/sctnet-b_8x4_160k_pets.py
bash tools/train.sh configs/sctnet/pets/sctnet-s_8x4_160k_pets.py
bash tools/train.sh configs/sctnet/pets/sctnet-s_nt_8x4_160k_pets.py

python tools/test.py configs/sctnet/pets/sctnet-b_8x4_160k_pets.py work_dirs/sctnet-b_8x4_160k_pets/lastese.pth --show-dir mIoU


python tools/analyze_logs.py work_dirs/sctnet-s_8x4_160k_pets瞎鸡儿指导/20250105_142438.log.json --keys mIoU mAcc aAcc --legend mIoU mAcc aAcc --out ./tran



# 
pip install "mmsegmentation==v0.26.0"
pip install yapf==0.40.1


python tools/train.py configs/sctnet/pets/sctnet-b_8x4_160k_pets.py
bash tools/train.sh configs/sctnet/pets/sctnet-b_8x4_160k_pets.py
bash tools/train.sh configs/sctnet/pets/sctnet-s_8x4_160k_pets.py

python tools/test.py configs/sctnet/pets/sctnet-b_8x4_160k_pets.py work_dirs/sctnet-b_8x4_160k_pets/lastese.pth --show-dir mIoU



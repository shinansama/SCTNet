pip install "mmsegmentation==v0.26.0"
pip install yapf==0.40.1



python tools/test.py configs/sctnet/cityscapes/sctnet-b_seg100_8x2_160k_cityscapes.py configs/sctnet/cityscapes/pretrain/SCTNet-B-Seg100.pth  --eval mIoU

python tools/test.py configs/sctnet/ADE20K/sctnet-b_8x4_160k_ade.py configs/sctnet/ADE20K/pretrain/SCTNet-B-ADE20K.pth  --eval mIoU

python tools/train.py configs/sctnet/pets/sctnet-b_8x4_160k_pets.py
bash tools/train.sh configs/sctnet/pets/sctnet-b_8x4_160k_pets.py


python tools/test.py configs/sctnet/pets/sctnet-b_8x4_160k_pets.py work_dirs/sctnet-b_8x4_160k_pets/lastese.pth --show-dir mIoU
bash tools/train.sh configs/sctnet/pets/sctnet-b_8x4_160k_pets.py



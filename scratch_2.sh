python tools/test.py configs\sctnet\cityscapes\sctnet-b_seg100_8x2_160k_cityscapes.py configs\sctnet\cityscapes\pretrain\SCTNet-B-Seg100.pth  --eval mIoU

python tools/test.py configs\sctnet\ADE20K\sctnet-b_8x4_160k_ade.py configs\sctnet\ADE20K\pretrain\SCTNet-B-ADE20K.pth  --eval mIoU

python tools/train.py configs\sctnet\pets\sctnet-b_8x4_160k_pets.py

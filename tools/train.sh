CONFIG=$1
GPUS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:2}

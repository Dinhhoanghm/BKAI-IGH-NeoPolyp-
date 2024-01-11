![Alt text](img/visualize.png)

## Setup

```bash
source ./script/setup.sh
```

## Training

```bash
source ./script/train.sh
```

## Testing

```bash
python -m neopolyp.infer --model [CHECKPOINT_PATH] --data_path [DATA_PATH] --save_path [SAVE_PATH]
```

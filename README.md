# Behavior Transformer

## Installation
Clone this repository and then run in the root directory:

```pip install -r requirements.txt```

## Usage
To launch a training job, from the root directory, run:

```python train_offline.py```

## Additional Setup
If you want to see metric logging, login to your weights & biases account using the information from [here.](https://docs.wandb.ai/quickstart)

You can then launch a logged training job by running:

```python train_offline.py --log```

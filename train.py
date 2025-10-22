# Now, it’s finally time to put all of the components you implemented together into your main training
#  script. It will pay off to make it easy to start training runs with different hyperparameters (e.g., by taking
#  them as command-line arguments), since you will be doing these many times later to study how different
#  choices impact training.
# Deliverable: Write a script that runs a training loop to train your model on user-provided input. In
#  particular, we recommend that your training script allow for (at least) the following:
#  • Ability to configure and control the various model and optimizer hyperparameters.
#  • Memory-efficient loading of large training and validation datasets with np.memmap.
#  • Serializing checkpoints to a user-provided path.
#  • Periodically logging training and validation performance (e.g., to console and/or an external
#  service like Weights & Biases)a.

import argparse
import os
import numpy as np
import torch

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.train_utils import get_batch, save_checkpoint, load_checkpoint
from cse599o_basics.model_utils import cross_entropy, lr_scheduler, gradient_clipping

def main():
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    # transformerLM hyperparameters
    parser.add_argument("--vocab_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)

    parser.add_argument("--num_steps", type=int, default=5000)

    # steps:
    # first, parse args
    args = parser.parse_args()

    # load dataset
    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # use apple silicon if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 

if __name__ == "__main__":
    main()
import argparse
import os
import numpy as np
import torch
import time

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.adamw import AdamW
from cse599o_basics.train_utils import get_batch, save_checkpoint, load_checkpoint
from cse599o_basics.model_utils import cross_entropy, softmax, lr_scheduler, gradient_clipping

def main():
    train_txt, valid_txt = "data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-valid.txt"
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    # transformerLM hyperparameters
    # parser.add_argument("--vocab_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # get_batch parameters
    parser.add_argument("--batch_size", type=int, default=32)

    # training parameters
    parser.add_argument("--num_steps", type=int, default=5000)

    args = parser.parse_args()
    context_length: int = args.context_length
    num_layers: int = args.num_layers
    d_model: int = args.d_model
    num_heads: int = args.num_heads
    d_ff: int = args.d_ff
    rope_theta: float = args.rope_theta
    batch_size: int = args.batch_size
    num_steps: int = args.num_steps

    print("Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab={}, merges=[])
    vocab_size = tokenizer.tokenizer.n_vocab

    print("Preparing datasets...")
    train_dataset, valid_dataset = prep_datasets(train_txt, valid_txt, tokenizer)

    print("Initializing model and optimizer...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta).to(device)
    optim = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    print("Starting training loop...")
    train(model, optim, train_dataset, valid_dataset, batch_size, context_length, num_steps, device)

    decode(model, tokenizer, optim, max_tokens=30, temperature=1.0, top_p=0.9, prompt="Once upon a time", ckpt_file="")

def prep_datasets(train_txt: str, valid_txt: str, tokenizer: BPETokenizer):
    if not os.path.exists("data/train_memmap.dat"):
        print("Tokenizing training data...")
        with open(train_txt, "r", encoding="utf-8") as f:
            train_data = f.read()
        train_tokens = tokenizer.encode(train_data)
        print("Creating training memmap dataset...")
        train_dataset = np.memmap("data/train_memmap.dat", dtype=np.uint16, mode="w+", shape=(len(train_tokens),))
        train_dataset[:] = np.array(train_tokens, dtype=np.uint16)
    if not os.path.exists("data/valid_memmap.dat"):
        print("Repeating for validation data...")
        with open(valid_txt, "r", encoding="utf-8") as f:
            valid_data = f.read()
        valid_tokens = tokenizer.encode(valid_data)
        valid_dataset = np.memmap("data/valid_memmap.dat", dtype=np.uint16, mode="w+", shape=(len(valid_tokens),))
        valid_dataset[:] = np.array(valid_tokens, dtype=np.uint16)

    train_dataset = np.memmap("data/train_memmap.dat", dtype=np.uint16, mode="r")
    valid_dataset = np.memmap("data/valid_memmap.dat", dtype=np.uint16, mode="r")
    return train_dataset, valid_dataset

def train(model: TransformerLM, optim: AdamW, 
          train_dataset: np.memmap, valid_dataset: np.memmap,
          batch_size, context_length, num_steps, device):
    # checkpoint file name: current timestamp (in progress work here)
    file_name = str(int(time.time()))
    # inputs, labels = get_batch(train_dataset, batch_size, context_length, device)
    for step in range(1, num_steps + 1):
        model.train()
        inputs, labels = get_batch(train_dataset, batch_size, context_length, device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        loss.backward()
        optim.step()

        if (num_steps < 50 and step % 2 == 0)  or (num_steps >= 50 and step % (num_steps // 50) == 0):
            print(f"Step {step}, Training Loss: {loss.item():.4f}")

        if (num_steps < 20 and step % 5 == 0) or (num_steps >= 20 and step % (num_steps // 10) == 0):
            model.eval()
            with torch.no_grad():
                val_inputs, val_labels = get_batch(valid_dataset, batch_size, context_length, device)
                val_outputs = model(val_inputs)
                val_loss = cross_entropy(val_outputs, val_labels)
                print(f"Step {step}, Validation Loss: {val_loss.item():.4f}")

            save_checkpoint(model, optim, step, os.path.join("checkpoints", f"checkpoint_step_{step}.pt"))
        

def decode(model: TransformerLM, tokenizer: BPETokenizer, optim: AdamW,
           max_tokens:int = 256, temperature: float = 1.0, top_p: float = 0.9, 
           prompt: str = "Once upon a time", ckpt_file: str = ""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if ckpt_file and os.path.exists(ckpt_file):
        print("Loading checkpoint...")
        load_checkpoint(ckpt_file, model, optim)
    else:
        print("No checkpoint. I sure hope you've trained this model!")

    print("Generating text...")
    model.eval()
    input_tokens = torch.tensor(tokenizer.encode(prompt)) # (seq_len,)
    input_tensor = input_tokens.unsqueeze(0).to(device) # (1, seq_len)
    eot = tokenizer.tokenizer.eot_token

    # keep list of generated tokens
    generated_tokens = input_tokens.tolist()

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_tensor) # (1, seq_len, vocab_size)
            last_token_softmax = softmax(outputs[0, -1, :] / temperature, dim=-1) # (vocab_size,)
            sorted_probs, sorted_indices = torch.sort(last_token_softmax, descending=True)
            sum = 0.0
            i = 0
            while sum < top_p and i < sorted_probs.size(0):
                sum += sorted_probs[i].item()
                i += 1
            top_p_indices = sorted_indices[:i]
            top_p_probs = sorted_probs[:i]
            top_p_probs = top_p_probs / top_p_probs.sum()

            index = torch.multinomial(top_p_probs, 1)
            next_token = top_p_indices[index] # (1,)
            generated_tokens.append(next_token.item())
            if next_token.item() == eot:
                break
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0).to(device)], dim=1)

    generated_text = tokenizer.decode(generated_tokens)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
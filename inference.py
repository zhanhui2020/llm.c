import torch
import tiktoken
import numpy as np

from train_gpt2 import GPT, GPTConfig

def read_wightdict(weights_dict, file, L, config):
    idx = 256
    with open(file, "rb") as f:
        data = f.read()
    data_copy = np.copy(data)
    weights = torch.from_numpy(np.frombuffer(data_copy, dtype=np.float32))
    weights_dict["transformer.wte.weight"] = weights[idx:idx+config["vocab_size"]*config["n_embd"]].reshape(config["vocab_size"], config["n_embd"])
    idx += config["vocab_size"]*config["n_embd"]
    weights_dict["transformer.wpe.weight"] = weights[idx:idx+config["block_size"]*config["n_embd"]].reshape(config["block_size"], config["n_embd"])
    idx += config["block_size"]*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.ln_1.weight"] = weights[idx:idx+config["n_embd"]]
        idx += config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.ln_1.bias"] = weights[idx:idx+config["n_embd"]]
        idx += config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.attn.c_attn.weight"] = weights[idx:idx+config["n_embd"]*3*config["n_embd"]].reshape(3*config["n_embd"], config["n_embd"])
        idx += 3*config["n_embd"]*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.attn.c_attn.bias"] = weights[idx:idx+3*config["n_embd"]]
        idx += 3*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.attn.c_proj.weight"] = weights[idx:idx+config["n_embd"]*config["n_embd"]].reshape(config["n_embd"], config["n_embd"])
        idx += config["n_embd"]*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.attn.c_proj.bias"] = weights[idx:idx+config["n_embd"]]
        idx += config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.ln_2.weight"] = weights[idx:idx+config["n_embd"]]
        idx += config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.ln_2.bias"] = weights[idx:idx+config["n_embd"]]
        idx += config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.mlp.c_fc.weight"] = weights[idx:idx+config["n_embd"]*4*config["n_embd"]].reshape(4*config["n_embd"], config["n_embd"])
        idx += 4*config["n_embd"]*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.mlp.c_fc.bias"] = weights[idx:idx+4*config["n_embd"]]
        idx += 4*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.mlp.c_proj.weight"] = weights[idx:idx+config["n_embd"]*4*config["n_embd"]].reshape(config["n_embd"], 4*config["n_embd"])
        idx += 4*config["n_embd"]*config["n_embd"]
    for i in range(L):
        weights_dict[f"transformer.h.{i}.mlp.c_proj.bias"] = weights[idx:idx+config["n_embd"]]
        idx += config["n_embd"]
    weights_dict["transformer.ln_f.weight"] = weights[idx:idx+config["n_embd"]]
    idx += config["n_embd"]
    weights_dict["transformer.ln_f.bias"] = weights[idx:idx+config["n_embd"]]
    return weights_dict

def load_model(file, model_type):
    config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    #model = GPT.from_pretrained("../openai-gpt2")
    model = GPT.from_pretrained("gpt2")
    weights = model.state_dict()
    weights = read_wightdict(weights, file, config_args['n_layer'], config_args)
    model.load_state_dict(weights, strict=True)
    state_dict = model.state_dict()
    print("="*40)
    print("=======GPT2 Architecture======")
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())
    print("="*40)
    return model

if __name__ == '__main__':
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    device = "cpu"
    max_new_tokens = 64
    temperature = 1.0
    top_k = 40
    model = load_model('gpt2_124M.bin', 'gpt2')
    model.eval()
    start = "hello, this is a toy demo for gpt2."
    while True:
        start = input("User:")

        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print("GPT2: " + decode(y[0].tolist()).lstrip(start))

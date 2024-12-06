import torch

def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

if __name__ == "__main__":
    clear_cuda_cache()
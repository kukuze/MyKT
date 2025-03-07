import torch
import numpy as np


def print_gpu_memory():
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print("-" * 50)


def test_memory_release():
    print("Initial GPU memory usage:")
    print_gpu_memory()

    # 创建一个2GB的张量并移到GPU
    size = int(2 * 1024 * 1024 * 1024 / 4)  # 2GB in float32
    tensor1 = torch.randn(size)
    tensor2 = torch.randn(size)

    # 将张量移到GPU
    print("\nMoving tensors to GPU...")
    tensor1 = tensor1.cuda()
    tensor2 = tensor2.cuda()
    print("GPU memory after allocation:")
    print_gpu_memory()

    # 创建类似history_info的元组
    history_info = (tensor1, tensor2)

    # 尝试释放内存
    print("\nTrying to free memory...")

    del history_info
    del tensor1
    del tensor2
    torch.cuda.empty_cache()
    print("GPU memory after attempted release:")
    print_gpu_memory()

    # 强制进行垃圾回收
    import gc
    gc.collect()

    print("\nGPU memory after garbage collection:")
    print_gpu_memory()


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_memory_release()
    else:
        print("CUDA is not available")
import torch

def native_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denomiator = torch.sum(numerator, dim=1)
    ret = numerator / denomiator[:, None]

    return ret

if __name__ == "__main__":
    x = torch.randn([32, 1024])
    ret = native_softmax(x)
    print("softmax output shape ", ret.shape)
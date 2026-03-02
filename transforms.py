'''
Various transforms of data for input pipelines
'''
import torch
import torchvision
import torchvision.transforms as transforms

# placing of the image channels transform
class SpreadChannelsTransform:
    """
    A PyTorch torchvision-compatible transform that spreads 3 color channels
    into a 3D volume with equal padding. Can be used in transforms.Compose().
    """
    def __init__(self, padding_size=2):
        self.padding_size = padding_size

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor image of size (C, H, W) to be spread.
                                   Typically outputted by transforms.ToTensor().
        Returns:
            torch.Tensor: Spaced Tensor of size (Depth, H, W)
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")

        c, h, w = tensor.shape
        if c != 3:
            raise ValueError(f"Input tensor must have exactly 3 channels (C, H, W). Got {c}.")

        if self.padding_size == 0:
            return tensor

        # 1. Create a block of zeros to insert between channels.
        # We ensure it matches the input tensor's dtype and device (e.g., GPU/CPU)
        zeros = torch.zeros((self.padding_size, h, w), dtype=tensor.dtype, device=tensor.device)

        # 2. Split the input tensor into its 3 individual channels, each of shape (1, H, W)
        c0, c1, c2 = torch.split(tensor, 1, dim=0)

        # 3. Concatenate the channels with the zero blocks interleaved in between
        volume = torch.cat((c0, zeros, c1, zeros, c2), dim=0)

        return volume

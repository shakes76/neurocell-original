'''
Unit test for placing color images into a cellular array in various ways

Made with assistance (Gemini Pro 3)
'''
import jax
import jax.numpy as jnp
import haiku as hk

#dataset
import torch
import torchvision
import torchvision.transforms as transforms

from universe import Universe3D #local
import utils #local
import visualize #local

utils.jax_status()

length = 32
channels = 3
num_classes = 10
batch_size = 1
layers = 8
color_layers = 3
padding = color_layers*2
depth_input = 2*padding+channels
centerx = (depth_input-1)//2
centery = length//2
centerz = length//2
print("padding", padding, "depth_input", depth_input)

# network IO paths
path = '/home/shakes/'
# path = 'F:/'

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

#===========
#transform preproc
#use x dim for depth since pytorch is channel first by default
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    SpreadChannelsTransform(padding_size=padding),
])

trainset = torchvision.datasets.CIFAR10(
    root=path+'data/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) #num_workers=6

# We draw an input to plot.
x, y = next(iter(train_loader))
print("x shape", x.shape)

# plot first batch
center_point = (centerx, centery, centerz)
visualize.plot_orthogonal_views(x[0], center=center_point)

# init
kernel_size = (9,9,9) #channel first
depth_full = 2*layers #-(channels-1) #(k-1)/2 hyperplanes per layer, counting from last channel
depth = depth_full//( (kernel_size[0]-1)//2 ) # propagation speed is (kernel_size-1)//2 per step, go upto first channel
Nx = depth_input + 1 + 2*(kernel_size[0]-1) #channel first
Ny = length + 2*(kernel_size[1]-1)
Nz = depth_full + length + 2*(kernel_size[2]-1)
print("N:", (Nx,Ny,Nz), "depth full:", depth_full, "Kernel:", kernel_size)

#setup network position, heatsink like setup
centerx = Nx//2
centery = Ny//2
centerz = Nz//2
#input
input_top = centerx-(depth_input-1)//2
input_left = centery-length//2
input_inner = centerz-(depth_full+length)//2
#output, doesn't have to be at end of layers, propagator will learn directions to output
output_top = centerx-num_classes//2 #include out layer
output_left = centery
output_inner = input_inner+(depth_full+length)+1 #include out layer #center-num_classes//2
# print("input_top", input_top, "input_left", input_left, "output_top", output_top, "output_left", output_left)
#setup parameters and lengths
input_size = (depth_input,length,length)
input_offset = (input_top,input_left,input_inner)
output_size = (num_classes,1,1)
output_offset = (output_top,output_left,output_inner)
activation = jax.nn.leaky_relu
final_activation = jax.nn.leaky_relu

#===========
##create model
def _forward(batch) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    mlp = Universe3D((Nx,Ny,Nz), sharpen=False)
    mlp.initialize_3D(input_size, output_size, input_offset, output_offset, depth, kernel_size, directions=1)
    return mlp(batch, activation, final_activation, norm=True)

# Make the network and optimizer. Haiku standard
network = hk.without_apply_rng(hk.transform_with_state(_forward))

# Initialize network and optimiser; note we draw an input to get shapes.
x, y = next(iter(train_loader))
x = jnp.asarray(x.numpy())
# x = jnp.transpose(x.numpy(), (0, 2, 3, 1)) #channel last
print("INIT SHAPE", x[0:1].shape)
params, state = network.init(jax.random.PRNGKey(42), x[0:1])
# print(params)

center_point = (centerx, centery, centerz)
# plt.imshow(params['universe3_d']['s'][:,center,:])
# plt.title("Initial States")
# plt.savefig("vnn_3D_classify_cifar_states_init.png", dpi=150)
# plt.show()
visualize.plot_orthogonal_views(params['universe3_d']['s'], center=center_point)

print("END")

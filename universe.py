"""
Universe object for Von Neumann networks
"""
from typing import Sequence, TypeAlias

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.scipy.signal as jsignal
import haiku as hk
import math

import filters #local

#kernels for larger networks
kernel_backward =  [[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]]
kernel_forward =   [[0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]]
STATE_ACTIVE = 0.7
STATE_SIGNAL = 0.2
STATE_ACTIVE_VALUE = 1.0
STATE_SIGNAL_VALUE = 0.0

class VectorCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting vertical/horizontal cells."""
    def __init__(self, in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=1.0, col=True):
        """
        Constructs a :class:`RandomNormal` like initializer with vector cells.

        in_sizes etc. can be single or a list. in_sizes and in_offsets must be same in number, likewise out_*
        """
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        self.in_offsets = in_offsets
        self.out_offsets = out_offsets
        self.is_states = is_states
        self.clear = clear
        self.random_init = random_init
        self.col = col
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")

    def clear_cells_vertical(self, cells, offset, after=False):
        """
        Clears cells vertically upto and including offset positions, assumes 2D
        """
        if after:
            result = cells.at[:,offset[1]:].set(0)  #clear cells after
        else:
            result = cells.at[:,:offset[1]+1].set(0) #clear cells before
        return result

    def clear_cells_horizontal(self, cells, offset, after=False):
        """
        Clears cells horizontally upto and including offset positions, assumes 2D
        """
        if after:
            result = cells.at[offset[0]:,:].set(0)  #clear cells after
        else:
            result = cells.at[:offset[0]+1,:].set(0) #clear cells before
        return result

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        if self.random_init > 0 or not self.random_init is None:
            cells = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            cells = jnp.zeros(shape, dtype=dtype)

        if self.is_states:
            #norm states to 0-1 range
            cells = jax.nn.sigmoid(cells)
            # cells = (cells - cells.min()) / (cells.max() - cells.min())

        if not isinstance(self.in_sizes, list):
            self.in_sizes = [self.in_sizes] #assume was one element, convert to list
            self.in_offsets = [self.in_offsets] #assume was one element, convert to list
        if not isinstance(self.out_sizes, list):
            self.out_sizes = [self.out_sizes] #assume was one element, convert to list
            self.out_offsets = [self.out_offsets] #assume was one element, convert to list

        for in_size, in_offset in zip(self.in_sizes, self.in_offsets):
            if self.is_states:
                # cells = jnp.ones(shape, dtype=dtype)
                inputs = jnp.full(in_size, STATE_ACTIVE_VALUE, dtype=dtype)
            else:
                # real_dtype = jnp.finfo(dtype).dtype
                # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
                # upper = 3.0
                # lower = -3.0
                # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
                # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
                inputs = jnp.ones(in_size, dtype=dtype)

            if self.clear:
                if self.col:
                    cells = self.clear_cells_vertical(cells, in_offset, after=False)
                else:
                    cells = self.clear_cells_horizontal(cells, in_offset, after=False)

            cells = jax.lax.dynamic_update_slice(cells, inputs, in_offset)

        for out_size, out_offset in zip(self.out_sizes, self.out_offsets):
            if self.is_states:
                # cells = jnp.ones(shape, dtype=dtype)
                outputs = jnp.full(out_size, STATE_ACTIVE_VALUE, dtype=dtype)
            else:
                # real_dtype = jnp.finfo(dtype).dtype
                # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
                # upper = 3.0
                # lower = -3.0
                # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
                # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
                outputs = jnp.ones(out_size, dtype=dtype)

            if self.clear:
                if self.col:
                    cells = self.clear_cells_vertical(cells, out_offset, after=True)
                else:
                    cells = self.clear_cells_horizontal(cells, out_offset, after=True)

            cells = jax.lax.dynamic_update_slice(cells, outputs, out_offset)

        return cells

class VerticalCells(VectorCells):
    """Initializes by sampling from a normal distribution and setting vertical cells."""
    def __init__(self, in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with vertical cells. See base class VectorCells.
        """
        super().__init__(in_sizes, out_sizes, in_offsets, out_offsets, is_states, clear, random_init, col=True)

class HorizontalCells(VectorCells):
    """Initializes by sampling from a normal distribution and setting horizontal cells."""
    def __init__(self, in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with horizontal cells. See base class VectorCells.
        """
        super().__init__(in_sizes, out_sizes, in_offsets, out_offsets, is_states, clear, random_init, col=False)

class SquareCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting open square (shape of a C) cells."""
    def __init__(self, in_size, out_size, in_offset, out_offset, is_states=False, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with vertical cells.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.is_states = is_states
        self.random_init = random_init
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        if self.is_states:
            # cells = jnp.ones(shape, dtype=dtype)
            inputs = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=dtype)
            outputs = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=dtype)
        else:
            # real_dtype = jnp.finfo(dtype).dtype
            # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
            # upper = 3.0
            # lower = -3.0
            # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
            # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
            inputs = jnp.ones(self.in_size, dtype=dtype)
            outputs = jnp.ones(self.out_size, dtype=dtype)

        if self.random_init > 0 or not self.random_init is None:
            cells = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            cells = jnp.zeros(shape, dtype=dtype)

        if self.is_states:
            #norm states to 0-1 range
            cells = jax.nn.sigmoid(cells)
            # cells = (cells - cells.min()) / (cells.max() - cells.min())

        length = self.in_size[0]

        #outputs
        cells = jax.lax.dynamic_update_slice(cells, outputs, self.out_offset)

        #inputs
        #top/up
        up = inputs[0:length//4].reshape(1,-1) #row vec
        # print("top:", up.shape, up)
        up = jnp.fliplr(up) #reverse
        shift = (self.in_offset[0], self.in_offset[1]+1) #one step right
        cells = jax.lax.dynamic_update_slice(cells, up, shift)
        #middle
        mid = inputs[length//4+1:length//2+length//4].reshape(-1,1) #col vec
        # print("mid:", mid.shape, mid)
        shift = (self.in_offset[0]+1, self.in_offset[1]) #one step down
        cells = jax.lax.dynamic_update_slice(cells, mid, shift)
        #bottom/down
        down = inputs[length//2+length//4+1:length//2+length//4+length//4].reshape(1,-1) #row vec
        # print("down:", down.shape, down)
        shift = tuple(map(lambda x, y: x+y, self.in_offset, mid.shape)) #move to bottom
        shift = (shift[0]+1, shift[1]) #one step down
        # print("shift:", shift)
        cells = jax.lax.dynamic_update_slice(cells, down, shift)

        return cells

class CoordinateCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting cells through 2D coordinates provided."""
    def __init__(self, in_coords, out_coords, out_offset, out_size, is_states=False, clear=True, random_init=1.0):
        """
        Constructs a :class:`RandomNormal` like initializer with cells with coordinates.
        """
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.out_offset = out_offset #unused
        self.out_size = out_size #unused
        self.is_states = is_states
        self.clear = clear
        self.random_init = random_init
        self.min_coord = in_coords[:, 1].min()
        self.max_coord = out_coords[:, 1].max()
        self.random_init = random_init
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")

    def clear_cells_outside(self, cells, coord, after=False):
        """
        Clears cells upto and including coords positions padded by 1
        """
        if after:
            result = cells.at[:,coord+1:].set(0)  #clear cells after
        else:
            result = cells.at[:,:coord].set(0) #clear cells before
        return result

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        n = len(self.in_coords)
        if self.is_states:
            # cells = jnp.ones(shape, dtype=dtype)
            inputs = STATE_ACTIVE_VALUE
            outputs = STATE_ACTIVE_VALUE
        else:
            # real_dtype = jnp.finfo(dtype).dtype
            # stddev = 1. / jnp.sqrt(max(shape[0], shape[1]))
            # upper = 3.0
            # lower = -3.0
            # cells = stddev * jax.random.truncated_normal(hk.next_rng_key(), lower, upper, shape, real_dtype)
            # cells = 1.0 * jax.random.normal(hk.next_rng_key(), shape)
            inputs = STATE_SIGNAL_VALUE
            outputs = STATE_SIGNAL_VALUE

        if self.random_init > 0 or not self.random_init is None:
            cells = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            cells = jnp.zeros(shape, dtype=dtype)

        if self.is_states:
            #norm states to 0-1 range
            cells = jax.nn.sigmoid(cells)
            # cells = (cells - cells.min()) / (cells.max() - cells.min())

        if self.clear:
            cells = self.clear_cells_outside(cells, self.min_coord, after=False)
            cells = self.clear_cells_outside(cells, self.max_coord, after=True)

        cells = cells.at[self.in_coords[:, 0], self.in_coords[:, 1]].set(inputs)
        cells = cells.at[self.out_coords[:, 0], self.out_coords[:, 1]].set(outputs)

        return cells

class FiltersCells(hk.initializers.Initializer):
    """Initializes by sampling from a normal distribution and setting filters."""
    def __init__(self, is_forward=True, random_init=1.0):
        self.is_forward = is_forward
        self.random_init = random_init
        # self.initializer = jax.nn.initializers.uniform(self.random_init)
        self.initializer = jax.nn.initializers.variance_scaling(self.random_init, mode="fan_in", distribution="truncated_normal")

    def __call__(self, shape: Sequence[int], dtype) -> jax.Array:
        # if not self.is_forward:
        #     nethood = filters.backward(dtype=dtype, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # else:
        #     nethood = filters.forward(dtype=dtype, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel

        # nethood = nethood + 0.1 * jax.random.normal(hk.next_rng_key(), nethood.shape)
        if self.random_init > 0 or not self.random_init is None:
            nethood = self.initializer(hk.next_rng_key(), shape, jnp.float32)
        else:
            nethood = jnp.ones(shape, dtype=dtype)

        return nethood

class UniverseLP(hk.Module):
    """
    Represents cellular structured universe of Von Neumann networks (VNNs) as a layered perceptron (LP)

    You must use the initialize_* members to create the type of layered perceptron, e.g. MLP etc. first
    Then call usual JAX/Haiku type networks calls that will call this classes implementation of the VNNs
    """
    def __init__(self, N, sharpen=True, dtype=jnp.float32, name=None):
        super().__init__(name=name)
        self.N = N
        self.grid_size = (N,N)
        self.sharpen = sharpen
        self.dtype = dtype
        self.states_dtype = jnp.float32
        # self.type_MLP = False
        # self.type_CLP = False
        #filters for spatial derivatives
        # self.nethood_backward = jnp.array(kernel_backward, dtype=dtype)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel, NHWC
        # self.nethood_forward = jnp.array(kernel_forward, dtype=dtype)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel, NHWC
        # self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.smooth_filter = filters.smooth(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.sharpen_filter = filters.sharpen(dtype=jnp.float32, norm=True)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.filters_shape = None
        self.img_shape = (1,N,N,1)
        self.window_stride = (1,1)
        self.forward_op = None

    def initialize(self, in_sizes, out_sizes, in_offsets, out_offsets, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells with a series of input and outputs as vectors, i.e. lists of vectors are provided, single values are also supported.

        Assumes inputs or outputs are of same length, different number of inputs and outputs can be provided.
        """
        self.in_size = in_sizes
        self.out_size = out_sizes
        self.in_offset = in_offsets
        self.out_offset = out_offsets
        self.kernel_size = kernel_size
        self.depth = depth
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(self.kernel_size[0]*self.kernel_size[1]*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=True, clear=False, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        # assumes all ins and outs are same size by taking first element, TODO: remove this restriction
        self.input_states = []
        for in_size in in_sizes:
            self.input_states.append(jnp.full(in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype))
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_full_mlp_list, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_mlp(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as a typical vertical set of layers like a traditional MLP
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        # self.depth = self.out_offset[1]-self.in_offset[1]
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(math.prod(self.kernel_size)*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = VerticalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=True, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = VerticalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=True, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_mlp_row(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as a typical vertical set of layers like a traditional MLP
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        # self.depth = self.out_offset[1]-self.in_offset[1]
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(math.prod(self.kernel_size)*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=True, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=True, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_mlp_row, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_slp(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3)):
        """
        Init all cells as an open square-like shape
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        # self.depth = self.out_offset[1]-self.in_offset[1]
        self.directions = 1
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        #init states (integers)
        #set input and output states
        self.s_init = SquareCells(in_size, out_size, in_offset, out_offset, is_states=True)
        # self.states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = None
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = None

        #init weights (random)
        self.w_init = SquareCells(in_size, out_size, in_offset, out_offset, is_states=False)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_full_slp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_clp(self, in_coords, out_coords, out_offset, out_size, layer_coords, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as an open circle-like shape
        """
        self.in_offset = in_coords
        self.out_coords = out_coords
        self.out_offset = out_offset
        self.out_size = out_size
        # self.layer_coords = layer_coords
        self.kernel_size = kernel_size
        self.depth = depth
        # print("depth:", depth)
        self.directions = directions

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        #init states (integers)
        self.s_init = CoordinateCells(self.in_offset, self.out_coords, out_offset, out_size, is_states=True, clear=False)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # self.states = states.at[self.layer_coords[:, 0], self.layer_coords[:, 1]].set(2)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = None
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = None

        #init weights (random)
        self.w_init = CoordinateCells(self.in_offset, self.out_coords, out_offset, out_size, is_states=False, clear=False)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_full_clp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_tm(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as a tape-like shape (horizontal lines)
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        self.directions = directions

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(self.kernel_size[0]*self.kernel_size[1]*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=False, random_init=init_scale)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=False, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_full_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_states(self):
        return self.states

    def get_filters(self):
        return self.filters

    def get_state(self):
        return hk.get_state("universe")

    # def forward_propagate(self, activate=nn.sigmoid, norm=True):
    #     self.universe = self.forward(self.universe, self.weights, self.biases, activate=activate, norm=norm)

    def __call__(self, inputs, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        # get params
        # states = self.states
        states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)
        biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #reset cells
        # cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        cells = hk.get_state("universe")
        cells = cells.at[...].set(0.)  #clear cells
        # hk.set_state("universe", cells)

        #smooth/sharpen weights
        if self.sharpen:
            weights = _filter(weights, self.sharpen_filter, self.dn, self.window_stride)

        #precompute FFTs of filters
        filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)
        # filters = filters[0,:,:,0]
        # length = self.kernel_size[0]
        # # FFTs of filter
        # filter_ffts = _fft_1D(self.N, length, filters)

        #set input
        # if self.type_MLP:
        #     cells = jax.lax.dynamic_update_slice(cells, inputs, self.in_offset)
        # else:
        #     cells = cells.at[self.in_coords[:, 0], self.in_coords[:, 1]].set(inputs)
        # hk.set_state("universe", cells)

        # force input/output states to be active
        if not self.input_states is None:
            if isinstance(self.input_states, list):
                for input_state, in_offset in zip(self.input_states, self.in_offset):
                    states = jax.lax.dynamic_update_slice(states, input_state, in_offset)
            else:
                states = jax.lax.dynamic_update_slice(states, self.input_states, self.in_offset)
        if not self.output_states is None:
            states = jax.lax.dynamic_update_slice(states, self.output_states, self.out_offset)
        if not self.output_weights is None:
            weights = jax.lax.dynamic_update_slice(weights, self.output_weights, self.out_offset)

        #forward
        # cells = _forward_propagate_fast(self.in_offset[1], self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn)
        # cells = _forward_propagate_fast_mlp(inputs, self.in_offset, self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn)
        # cells = _forward_propagate_fast_1D(self.in_offset[1], self.depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm)
        # forward_propagate_fast = jax.vmap(_forward_propagate_fast_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None])
        cells, outputs = self.forward_op(inputs, self.in_offset, self.out_offset, self.out_size, self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn, self.window_stride)
        hk.set_state("universe", cells[0]) #remove vmap batch dim

        #read output
        # if self.type_MLP:
        #     return jax.lax.dynamic_slice(cells, self.out_offset, self.out_size)
        # else:
        #     # value = cells[self.out_coords[:, 0], self.out_coords[:, 1]]
        #     # print("out:", value)
        #     # return value
        #     return cells[self.out_coords[:, 0], self.out_coords[:, 1]]
        return outputs

    def save(self, model_name):
        jnp.savez(model_name, weights=self.weights, biases=self.biases, states=self.states, allow_pickle=False)

    def load(self, filename):
        with jnp.load(filename, allow_pickle=False) as data:
            self.weights = data['weights']
            self.biases = data['biases']
            self.states = data['states']
            self.filters = data['filters']
        self.N = self.weights.shape[0]
        self.grid_size = self.weights.shape
        self.dtype = self.weights.dtype
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)

class UniverseTM(UniverseLP):
    """
    Represents cellular structured universe of Von Neumann networks (VNNs) as a Turing machine (TM) pure 1D tape model.

    You must use the initialize_* members to create the type of layered perceptron, e.g. MLP etc. first
    Then call usual JAX/Haiku type networks calls that will call this classes implementation of the VNNs
    """
    def __init__(self, N, sharpen=True, dtype=jnp.float32, name=None):
        super().__init__(N, sharpen=sharpen, dtype=dtype, name=name)
        self.grid_size = (1,self.N)
        self.filters_shape = None
        self.img_shape = (1,1,N,1)

    def initialize(self, in_size, out_size, in_offset, out_offset, depth, kernel_size=(3,3), directions=1):
        """
        Init all cells as a typical vertical set of layers like a traditional MLP
        """
        self.in_size = in_size
        self.out_size = out_size
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.kernel_size = kernel_size
        self.depth = depth
        # self.depth = self.out_offset[1]-self.in_offset[1]
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(self.kernel_size[0]*self.kernel_size[1]) #shape is (rows,cols) before array creation

        #init states
        self.s_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=True, clear=False, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        self.input_states = jnp.full(self.in_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = HorizontalCells(in_size, out_size, in_offset, out_offset, is_states=False, clear=False, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWC', 'IHWO', 'NHWC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_tape, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

class Universe:
    """
    Represents cellular structured universe of Von Neumann networks
    """
    def __init__(self, N, dtype=jnp.float32):
        self.epsilon = 1e-6
        self.N = N
        self.grid_size = (N,N)
        self.dtype = dtype
        self.states_dtype = jnp.uint8
        self.key = jax.random.key(42)
        #filters for spatial derivatives
        #self.nethood_backward = jnp.array(kernel_backward, dtype=dtype) # 8 connected kernel
        #self.nethood_forward = jnp.array(kernel_forward, dtype=dtype) # 8 connected kernel
        self.nethood_backward = filters.backward(dtype=jnp.float32, norm=False) # 8 connected kernel
        self.nethood_forward = filters.forward(dtype=jnp.float32, norm=False) # 8 connected kernel
        # print("filter:", self.nethood_forward)
        self.initialize() #universe and prev_universe

    def initialize(self):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)
        self.states = jnp.zeros(self.grid_size, dtype=self.states_dtype)

        new_key, subkey = jax.random.split(self.key)
        self.weights = 1.0 * jax.random.normal(subkey, self.grid_size, dtype=self.dtype)
        self.key = new_key

        self.save_state()

    def initialize_vertical(self, in_size, out_size, in_offset=None, out_offset=None):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)
        self.states = jnp.ones(self.grid_size, dtype=self.states_dtype)

        new_key, subkey = jax.random.split(self.key)
        self.weights = 1.0 * jax.random.normal(subkey, self.grid_size)
        self.biases = jnp.zeros(self.grid_size, dtype=self.dtype)
        self.key = new_key

        #setup in/out weights
        self.in_offset = (self.N//4, self.N//8)
        if in_offset:
            self.in_offset = in_offset
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if out_offset:
            self.out_offset = out_offset
        in_passives = jnp.full(in_size, 2, dtype=self.states_dtype)
        out_passives = jnp.full(in_size, 2, dtype=self.states_dtype)
        inputs = jnp.ones(in_size, dtype=self.weights.dtype)
        outputs = jnp.ones(out_size, dtype=self.weights.dtype)
        weights = self.clear_cells_vertical(self.weights, self.in_offset, after=False)
        self.weights = jax.lax.dynamic_update_slice(weights, inputs, self.in_offset)
        states = self.clear_cells_vertical(self.states, self.in_offset, after=False)
        self.states = jax.lax.dynamic_update_slice(states, in_passives, self.in_offset)
        weights = self.clear_cells_vertical(self.weights, self.out_offset, after=True)
        self.weights = jax.lax.dynamic_update_slice(weights, outputs, self.out_offset)
        states = self.clear_cells_vertical(self.states, self.out_offset, after=True)
        self.states = jax.lax.dynamic_update_slice(states, out_passives, self.out_offset)

        self.save_state()

    def get_cells(self):
        return self.universe

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_states(self):
        return self.states

    def save_state(self):
        self.previous_universe = self.universe

    def restore_state(self):
        self.universe = self.previous_universe

    def reset_state(self):
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)

    def clear_cells_vertical(self, universe, offset, after=False):
        """
        Clears cells upto and including offset positions
        """
        if after:
            result = universe.at[:,offset[1]:].set(0)  #clear cells after
        else:
            result = universe.at[:,:offset[1]+1].set(0) #clear cells before
        return result

    def set_input_vector(self, inputs, offset=None):
        """
        Set in vector at address with size
        """
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        cells = self.clear_cells_vertical(self.universe, self.in_offset, after=False)
        self.universe = jax.lax.dynamic_update_slice(cells, inputs, self.in_offset)
        # weights = self.clear_cells_vertical(self.weights, self.in_offset, after=False)
        # self.weights = jax.lax.dynamic_update_slice(weights, inputs, self.in_offset)

    def set_output_vector(self, outputs, offset=None):
        """
        Write out vector from address with size
        """
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        cells = self.clear_cells_vertical(self.universe, self.out_offset, after=True)
        self.universe = jax.lax.dynamic_update_slice(cells, outputs, self.out_offset)
        # weights = self.clear_cells_vertical(self.weights, self.out_offset, after=True)
        # self.weights = jax.lax.dynamic_update_slice(weights, outputs, self.out_offset)

    def set_layer_vertical(self, layer_size, offset=None):
        passives = jnp.full(layer_size, 2, dtype=self.states_dtype)
        self.states = jax.lax.dynamic_update_slice(self.states, passives, offset)

    def get_input_vector(self, input_size, offset=None):
        """
        Read in vector from address with size
        """
        self.in_offset = (self.N//4, self.N//8)
        if offset:
            self.in_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.in_offset, input_size)

    def get_output_vector(self, output_size, offset=None):
        """
        Read out vector from address with size
        """
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if offset:
            self.out_offset = offset
        return jax.lax.dynamic_slice(self.universe, self.out_offset, output_size)

    def shift(self, value):
        self.universe = self.universe - value #account for sigmoid etc.

    def forward(self, cells, weights, biases, activate=nn.sigmoid, norm=True):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_forward, mode='same')
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #check states
        #if passive apply activation, else signal state so weight
        cells = jnp.where(self.states==1, cells*weights, cells) #weight if signal
        cells = jnp.where(jnp.logical_and(self.states==2,cells!=0), activate(cells+biases), cells) #activate if passive
        return cells

    def forward_propagate(self, activate=nn.sigmoid, norm=True):
        self.universe = self.forward(self.universe, self.weights, self.biases, activate=activate, norm=norm)

    def backward(self, cells, activate=nn.sigmoid, norm=True):
        #get weighted sum of neighbors using convolution
        cells = jsignal.convolve2d(cells, self.nethood_backward, mode='same')
        #apply weights
        cells = cells * self.weights
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #excite and return
        return activate(cells)
        '''
        #run the network in reverse
        #excite
        cells = activate(cells)
        if norm:
            #norm
            cells = jax.nn.standardize(cells, axis=(0,1))
        #apply weights
        cells = cells / (self.weights+1e-8)
        #get weighted sum of neighbors using convolution
        return jsignal.convolve2d(cells, self.nethood_backward, mode='same')'''

    def backward_propagate(self, activate=nn.sigmoid, norm=True):
        #excite and return
        self.universe = self.backward(self.universe, activate=activate, norm=norm)

    def loss(self, cells, depth, labels, inputs, in_offset, output_size, out_offset, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        """
        Cross entropy
        """
        #set input
        # cells = self.clear_cells_vertical(cells, in_offset, after=False)
        cells = jax.lax.dynamic_update_slice(cells, inputs, in_offset)
        #forward
        for i in range(1, depth):
            cells = self.forward(cells, activate=activate, norm=norm)
        cells = self.forward(cells, activate=final_activate, norm=norm)
        #read output
        logits = jax.lax.dynamic_slice(cells, out_offset, output_size)
        return -1 * jnp.mean(labels * jnp.log(logits + self.epsilon) + (1. - labels) * jnp.log(1. - logits + self.epsilon)) #CE
        # return jnp.mean((labels - logits)**2) #MSE

    def loss_perturb(self, params, deltas, labels, inputs, in_offset, output_size, out_offset, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        """
        Cross entropy
        We use the perturbation trick, so get the layer by layer grad values for the overall loss
        See https://github.com/google/jax/discussions/5336#discussioncomment-269983
        """
        depth = len(deltas)
        weights, biases = params
        #set input
        # cells = self.clear_cells_vertical(cells, in_offset, after=False)
        cells = jax.lax.dynamic_update_slice(self.universe, inputs, in_offset)
        #forward
        for i in range(1, depth):
            cells = self.forward(cells, weights, biases, activate=activate, norm=norm)
            # cells = cells + deltas[i-1] #have grad return layer grad
        cells = self.forward(cells, weights, biases, activate=final_activate, norm=norm)
        # cells = cells + deltas[depth-1]
        # print("activations size:", len(activations))

        #read output
        logits = jax.lax.dynamic_slice(cells, out_offset, output_size)
        loss_value = -1 * jnp.mean(labels * jnp.log(logits + self.epsilon) + (1. - labels) * jnp.log(1. - logits + self.epsilon)) #CE

        return loss_value

    def value_and_grad(self, depth, labels, inputs, in_offset, out_size, out_offset):
        return jax.value_and_grad(self.loss)(self.universe, depth, labels, inputs, in_offset, out_size, out_offset, norm=False)

    def value_and_grad_perturb(self, depth, labels, inputs, in_offset, out_size, out_offset):
        """
        We use delta values to perturb the grad of the loss function
        Setting delta to zero means the grad will return the gradient of that layer
        """
        deltas = []
        for i in range(1,depth+1):
            deltas.append(0.0)
        deltas = tuple(deltas)
        # print("deltas:", deltas)
        params = [self.weights, self.biases]
        return jax.value_and_grad(self.loss_perturb)(params, deltas, labels, inputs, in_offset, out_size, out_offset, norm=False)

    def value_and_grad_update(self, lr, depth, labels, inputs, in_offset, out_size, out_offset):
        self.loss_value, grads_output = self.value_and_grad_perturb(depth, labels, inputs, in_offset, out_size, out_offset)
        # print("loss_value", self.loss_value)
        # print("layer grads", len(self.layer_grads), self.layer_grads[0].shape)
        # print("grads", len(self.grads_output), self.grads_output)
        # print("grads", len(self.grads_output[0]), self.grads_output[0])
        # print("grads", len(self.grads_output[1]), self.grads_output[1])

        #compose per layer grads
        self.grads = grads_output[0]
        self.bias_grads = grads_output[1]
        # self.grads = jnp.zeros_like(self.universe)
        # for i in range(1,depth):
        #     self.grads = self.grads.at[:,self.out_offset[1]-i].set(self.layer_grads[depth-i-1][:,self.out_offset[1]-i])
        # self.grads = self.grads.at[:,self.in_offset[1]].set(self.grads_output[0][:,self.in_offset[1]])

        #apply corrections
        self.weights = self.weights - lr*self.grads
        self.biases = self.biases - lr*self.bias_grads

        return self.loss_value, self.grads, self.bias_grads

    def gradient_update(self, lr, step, depth, inputs, input_size, in_offset=None, out_offset=None, activate=nn.sigmoid):
        """
        Computes the gradient based on the step and backprops it
        Computes and applies the updates to the saved state, assigns to current state
        """
        self.out_offset = (self.N//4+self.N//8, self.N//8+self.N//8)
        if out_offset:
            self.out_offset = out_offset

        #tracking changes
        updates = jnp.zeros_like(self.universe)
        #write step and backprop to get updates to weights required
        # updates = jax.lax.dynamic_update_slice(updates, step, self.out_offset) #include out

        #write step and backprop to get updates to weights required
        cells = self.clear_cells_vertical(self.universe, self.out_offset, after=True)
        cells = jax.lax.dynamic_update_slice(cells, step, self.out_offset)
        # cells = nn.sigmoid(cells)
        #apply weights
        # cells = cells / (self.weights+1e-8)
        updates = updates.at[:,self.out_offset[1]].set(cells[:,self.out_offset[1]])

        cells = self.backward(cells, activate=activate, norm=False)
        updates = updates.at[:,self.out_offset[1]-1].set(cells[:,self.out_offset[1]-1])
        for i in range(2,depth+1): #+1 includes in
            cells = self.backward(cells, activate=activate, norm=False)
            # cells = cells - 0.5 #account for sigmoid etc.
            #track updated weights per 'layer'
            updates = updates.at[:,self.out_offset[1]-i].set(cells[:,self.out_offset[1]-i])
            # self.universe = self.universe + cells
        # print("backprop:", cells)
        #copy updates to universe
        self.universe = cells.at[:,(self.out_offset[1]-depth):(self.out_offset[1]+1)].set(updates[:,(self.out_offset[1]-depth):(self.out_offset[1]+1)])

        #rescale by predicted input
        #take lr step
        # self.universe = lr*self.universe
        #rescale changes to be normalised
        # self.universe = jax.nn.standardize(self.universe, axis=(0,1))

        #L2 norm of weights
        # jnp.sum(self.universe)
        # self.universe = self.universe/jnp.abs(self.universe)

        #apply corrections
        # scaling = 1.0
        self.weights = self.weights - lr*self.universe
        # self.weights = jax.nn.standardize(self.weights, axis=(0,1))
        #restore input (data consistency)
        # self.universe = jax.lax.dynamic_update_slice(cells, X[0].reshape(-1,1), input_offset)

    def save(self, model_name):
        jnp.savez(model_name, weights=self.weights, biases=self.biases, states=self.states, allow_pickle=False)

    def load(self, filename):
        with jnp.load(filename, allow_pickle=False) as data:
            self.weights = data['weights']
            self.biases = data['biases']
            self.states = data['states']
        self.N = self.weights.shape[0]
        self.grid_size = self.weights.shape
        self.dtype = self.weights.dtype
        self.universe = jnp.zeros(self.grid_size, dtype=self.dtype)

class Universe3D(UniverseLP):
    """
    For 3D MLP-like networks to support imaging etc.
    """
    def __init__(self, N, sharpen=True, dtype=jnp.float32, name=None):
        super().__init__(N, sharpen=sharpen, dtype=dtype, name=name)
        self.grid_size = (N,N,N)
        self.img_shape = (1,N,N,N,1)
        self.window_stride= (1,1,1)
        self.sharpen_filter = filters.sharpen_3D(dtype=jnp.float32, norm=True)[jnp.newaxis,...,jnp.newaxis] # 8 connected kernel

    def initialize_3D(self, in_sizes, out_sizes, in_offsets, out_offsets, depth, kernel_size=(3,3,3), directions=1):
        """
        Init all cells with a series of input and outputs as vectors, i.e. lists of vectors are provided, single values are also supported.

        Assumes inputs or outputs are of same length, different number of inputs and outputs can be provided.

        Uses full 3D convs as the forward operator, expensive.
        """
        self.in_size = in_sizes
        self.out_size = out_sizes
        self.in_offset = in_offsets
        self.out_offset = out_offsets
        self.kernel_size = kernel_size
        self.depth = depth
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        # init_scale = 1./jnp.sqrt(self.depth*self.N*self.N)
        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=True, clear=False, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        # assumes all ins and outs are same size by taking first element, TODO: remove this restriction
        self.input_states = jnp.full(in_sizes, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],kernel_size[2],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWDC', 'IHWDO', 'NHWDC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_full_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def initialize_hyperplane(self, in_sizes, out_sizes, in_offsets, out_offsets, depth, kernel_size=(3,3,3), directions=1):
        """
        Init all cells with a series of input and outputs as vectors, i.e. lists of vectors are provided, single values are also supported.

        Assumes inputs or outputs are of same length, different number of inputs and outputs can be provided.

        Uses hyperplanes (n-1 D) convs per layer as the forward operator, more efficient.
        """
        self.in_size = in_sizes
        self.out_size = out_sizes
        self.in_offset = in_offsets
        self.out_offset = out_offsets
        self.kernel_size = kernel_size
        self.depth = depth
        self.directions = directions
        # print("depth:", self.depth)

        cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        hk.set_state("universe", cells)

        # init_scale = 1./jnp.sqrt(self.depth*self.N*self.N)
        init_scale = 1./jnp.sqrt(self.depth*self.N)
        init_filter_scale = 1./jnp.sqrt(self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]*self.directions) #shape is (rows,cols) before array creation

        #init states
        self.s_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=True, clear=False, random_init=init_scale)
        # states = self.s_init(shape=self.grid_size, dtype=self.states_dtype)
        # passives = jnp.full(layer_size, STATE_ACTIVE, dtype=self.states_dtype)
        # self.states = jax.lax.dynamic_update_slice(states, passives, layer_offset)
        self.states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        # print("States:", self.states)
        # hk.set_state("states", self.states)
        # we need input and out shaped array to force input/output states to be active, used later
        # assumes all ins and outs are same size by taking first element, TODO: remove this restriction
        self.input_states = jnp.full(in_sizes, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_states = jnp.full(self.out_size, STATE_ACTIVE_VALUE, dtype=self.states_dtype)
        self.output_weights = jnp.full(self.out_size, 1., dtype=jnp.float32)

        #init weights (random)
        self.w_init = VectorCells(in_sizes, out_sizes, in_offsets, out_offsets, is_states=False, clear=False, random_init=init_scale)
        self.weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)

        #init biases
        self.b_init = jnp.zeros
        self.biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #init filters
        self.filters_shape = (1,kernel_size[0],kernel_size[1],kernel_size[2],self.directions)
        self.f_init = FiltersCells(is_forward=False, random_init=init_filter_scale) # 8 connected kernel
        # nethood_forward = filters.forward(dtype=jnp.float32, norm=False)[jnp.newaxis,:,:,jnp.newaxis] # 8 connected kernel
        self.filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)

        self.dn = jax.lax.conv_dimension_numbers(self.img_shape,     # only ndim matters, not shape
                                self.filters.shape,  # only ndim matters, not shape
                                ('NHWDC', 'IHWDO', 'NHWDC'))  # the important bit

        self.forward_op = jax.vmap(_forward_propagate_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None, None])

        return hk.get_state("universe")

    def __call__(self, inputs, activate=nn.tanh, final_activate=nn.sigmoid, norm=True):
        # get params
        # states = self.states
        states = hk.get_parameter("s", shape=self.grid_size, dtype=self.states_dtype, init=self.s_init)
        weights = hk.get_parameter("w", shape=self.grid_size, dtype=self.dtype, init=self.w_init)
        biases = hk.get_parameter("b", shape=self.grid_size, dtype=self.dtype, init=self.b_init)

        #reset cells
        # cells = jnp.zeros(self.grid_size, dtype=self.dtype)
        cells = hk.get_state("universe")
        cells = cells.at[...].set(0.)  #clear cells
        # hk.set_state("universe", cells)

        #smooth/sharpen weights
        if self.sharpen:
            weights = _filter(weights, self.sharpen_filter, self.dn, self.window_stride)

        #precompute FFTs of filters
        filters = hk.get_parameter("f", shape=self.filters_shape, dtype=self.dtype, init=self.f_init)
        # filters = filters[0,:,:,:,0]

        # force input/output states to be active
        if not self.input_states is None:
            if isinstance(self.input_states, list):
                for input_state, in_offset in zip(self.input_states, self.in_offset):
                    states = jax.lax.dynamic_update_slice(states, input_state, in_offset)
            else:
                states = jax.lax.dynamic_update_slice(states, self.input_states, self.in_offset)
        states = jax.lax.dynamic_update_slice(states, self.output_states, self.out_offset)
        if not self.output_weights is None:
            weights = jax.lax.dynamic_update_slice(weights, self.output_weights, self.out_offset)

        #forward
        # cells = _forward_propagate_fast(self.in_offset[1], self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn)
        # cells = _forward_propagate_fast_mlp(inputs, self.in_offset, self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn)
        # cells = _forward_propagate_fast_1D(self.in_offset[1], self.depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm)
        # forward_propagate_fast = jax.vmap(_forward_propagate_fast_mlp, in_axes=[0, None, None, None, None, None, None, None, None, None, None, None, None, None])
        cells, outputs = self.forward_op(inputs, self.in_offset, self.out_offset, self.out_size, self.depth, cells, weights, biases, states, filters, activate, final_activate, norm, self.dn, self.window_stride)
        hk.set_state("universe", cells[0]) #remove vmap batch dim

        return outputs

#JIT compiled members for performance
from functools import partial

@partial(jax.jit, static_argnums=(0,1,7,8,9,10))
def _forward_propagate(index, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm, dn):
    """
    Propagate input across cellular net based on weights etc. layer by layer in a column based structure
    """
    #propagate
    for i in range(1, depth):
        result_block = _forward(index+i, cells, weights, biases, states, filter_ffts, activate, norm, dn)
        cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index+i))
    result_block = _forward(index+depth, cells, weights, biases, states, filter_ffts, final_activate, norm, dn)
    return jax.lax.dynamic_update_slice(cells, result_block, (0,index+depth))

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13))
def _forward_propagate_full(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm, dn):
    """
    Propagate input across cellular net based on weights etc. for all layers across entire universe
    """
    #place inputs
    cells = jax.lax.dynamic_update_slice(cells, input, offset)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(cells, weights, biases, states, filter_ffts, activate, norm, dn) #not inplace
        # cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index+i)) #write result for next propagate
    cells = _forward_full(cells, weights, biases, states, filter_ffts, final_activate, norm, dn) #final prop and activate
    # cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index+depth))

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13))
def _forward_propagate_full_slp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm, dn):
    """
    Propagate input across cellular net based on weights etc. for all layers across entire universe
    """
    width = out_offset[1]-offset[1] #to compute in subregion of the nets
    length = input.shape[0]

    #place inputs
    #top/up
    up = input[0:length//4].reshape(1,-1) #row vec
    # print("top:", up.shape, up)
    up = jnp.fliplr(up) #revrse
    shift = (offset[0], offset[1]+1) #one step right
    cells = jax.lax.dynamic_update_slice(cells, up, shift)
    #middle
    mid = input[length//4+1:length//2+length//4].reshape(-1,1) #col vec
    # print("mid:", mid.shape, mid)
    shift = (offset[0]+1, offset[1]) #one step down
    cells = jax.lax.dynamic_update_slice(cells, mid, shift)
    #bottom/down
    down = input[length//2+length//4+1:length//2+length//4+length//4].reshape(1,-1) #row vec
    # print("down:", down.shape, down)
    shift = tuple(map(lambda x, y: x+y, offset, mid.shape)) #move to bottom
    shift = (shift[0]+1, shift[1]) #one step down
    # print("shift:", shift)
    cells = jax.lax.dynamic_update_slice(cells, down, shift)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(width, cells, weights, biases, states, filter_ffts, activate, norm, dn) #not inplace
        # cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index+i)) #write result for next propagate
    cells = _forward_full(width, cells, weights, biases, states, filter_ffts, final_activate, norm, dn) #final prop and activate
    # cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index+depth))

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(2,3,4,10,11,12,13))
def _forward_propagate_full_clp(input, coords, out_offset, out_size, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm, dn):
    """
    Propagate input across cellular net based on weights etc. for all layers across entire universe
    """
    width = 0 #unused
    # length = input.shape[0]

    cells = cells.at[coords[:, 0], coords[:, 1]].set(input)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(width, cells, weights, biases, states, filter_ffts, activate, norm, dn) #not inplace
    cells = _forward_full(width, cells, weights, biases, states, filter_ffts, final_activate, norm, dn) #final prop and activate

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(0,1,7,8,9))
def _forward_propagate_fast_1D(index, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm):
    """
    Propagate input across cellular net based on weights etc. layer by layer in a column based structure

    Uses Fourier convolution property
    """
    #propagate
    for i in range(1, depth):
        result_block = _forward_fast_1D(index+i, cells, weights, biases, states, filter_ffts, activate, norm)
        cells = jax.lax.dynamic_update_slice(cells, result_block, (0,index+i))
    result_block = _forward_fast_1D(index+depth, cells, weights, biases, states, filter_ffts, final_activate, norm)
    return jax.lax.dynamic_update_slice(cells, result_block, (0,index+depth))

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _forward_propagate_mlp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses column/last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    """
    zeros_tuple = (0,) * (len(offset)-1) #offset for entire hyper plane

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-1]+1 #first col (or last hyper plane) next to input
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total
        result_block = _forward_column(index+i, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i,)) #place back into col (or hyper plane)
    # print("forward index final",index+depth)
    result_block = _forward_column(index+depth, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+depth,)) #place back into final col (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _forward_propagate_mlp_row(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses row/second-last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    """
    zeros_tuple = (0,) * (len(offset)-2) #offset for entire hyper plane upto second last (row)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-2]+1 #first row (or second-last hyper plane) next to input
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total
        result_block = _forward_row(index+i, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i,)+(0,)) #place back into row (or hyper plane)
    # print("forward index final",index+depth)
    result_block = _forward_row(index+depth, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+depth,)+(0,)) #place back into final row (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _forward_propagate_tape(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a Turing machine like tape.

    Uses 1D computations for efficiency and propagates up to provided depth.
    """
    zeros_tuple = (0,) * (len(offset)-2) #offset for entire hyper plane upto second last (row)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-2] #first and only row, all computations to be done in 1D, no layers
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total, i unused as there are no layers
        cells = _forward_row(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        # cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index,)+(0,)) #place back into row (or hyper plane)
    # print("forward index final",index+depth)
    cells = _forward_row(index, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer
    # cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index,)+(0,)) #place back into final row (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _forward_propagate_full_mlp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP
    """
    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    for i in range(1, depth):
        cells = _forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _forward_propagate_full_mlp_list(inputs, offsets, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP, inputs are list of inputs, one output still
    """
    for input, offset in zip(inputs, offsets): #place inputs to cells
        cells = jax.lax.dynamic_update_slice(cells, input, offset)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

# forward step operators
@partial(jax.jit, static_argnums=(5,6,7))
def _forward(cells, weights, biases, states, nethood, activate, norm, dn):
    """
    Compute the forward pass by computing each neuron depending on its state.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,:,:,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        (1,1),  # window strides
                                        'SAME', # padding mode
                                        (1,1),  # lhs/image dilation
                                        (1,1),  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    cells = cells[0,:,:,0] # NHWC

    if norm:
        #norm
        cells = jax.nn.standardize(cells, axis=(0,1))

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight
    cells = jnp.where(jnp.logical_and(states>=STATE_SIGNAL,states<STATE_ACTIVE), cells*weights, cells) #weight if signal
    if activate == None:
        cells = jnp.where(states>=STATE_ACTIVE, cells+biases, cells) #activate if passive, more accurate
    else:
        cells = jnp.where(states>=STATE_ACTIVE, activate(cells+biases), cells) #activate if passive, more accurate
    # cells = jnp.where(states==2, activate(cells+biases), cells) #activate if passive, faster

    return cells

@partial(jax.jit, static_argnums=(0,6,7,8,9))
def _forward_column(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1)):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise
    """
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    width = filter_shape[-1] # last dim, column in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-1])) # half of other dims of filter
    height_pad_shape = height_pad_shape + (0,) + (0,) #padding for each dim plus nth dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = max(1, (width-1)//2) # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at last dim
    # sub_cells = cells[:,index-1:index-1+width]
    sub_weights = weights[...,index]
    sub_biases = biases[...,index]
    sub_states = states[...,index]
    # print("sub_states",sub_states.shape)

    #extract sub array of filter width to conv
    # print("cells",cells.shape)
    result_block = cells[...,index-width_pad:index-width_pad+width]

    #restore to nD
    result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
    sub_weights = sub_weights[...,jnp.newaxis,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel

    # print("result_block",result_block.shape, "filters", filters.shape)
    #valid mode approach in all dims, reduced complexity by deliberately squishing last hyperplane as well to current index
    #handles boundaries better?
    result_block = _fast_conv(result_block, filters, dn, window_strides, padding_mode="VALID")
    # print("result_block conv",result_block.shape)
    result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #Same mode approach
    # result_block = _fast_conv(result_block, filters, dn, padding_mode="SAME")
    # result_block = result_block[:,(width_pad+1)-1].reshape(-1,1) #reshape promotes to column vec, -1 for 0 indexing
    # print("result_block padded",result_block.shape)

    if norm:
        #norm
        result_block = jax.nn.standardize(result_block, axis=(0,1))

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    return jnp.mean(result_block, axis=-1) # averaged across filters, back to 1 channel dim

@partial(jax.jit, static_argnums=(0,6,7,8,9))
def _forward_row(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1)):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise
    """
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    width = filter_shape[-2] # second-last dim, row in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-2])) # half of other dims of filter, no padding in second last dim
    height_pad_shape = height_pad_shape + (0,) + ((filter_shape[-1]-1)//2,) + (0,) #padding for each dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = (width-1)//2 # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at second-last dim
    # sub_cells = cells[:,index-1:index-1+width]
    sub_weights = weights[...,index,:]
    sub_biases = biases[...,index,:]
    sub_states = states[...,index,:]
    # print("sub_states",sub_states.shape)

    #extract sub array of filter width to conv
    # print("cells",cells.shape)
    result_block = cells[...,index-width_pad:index-width_pad+width,:]
    # print("result_block",result_block.shape)

    #restore to nD
    result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
    sub_weights = sub_weights[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel

    # print("result_block",result_block.shape, "filters", filters.shape)
    #valid mode approach, reduced complexity, handles boundaries better?
    result_block = _fast_conv(result_block, filters, dn, window_strides, padding_mode="VALID")
    result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #Same mode approach
    # result_block = _fast_conv(result_block, filters, dn, padding_mode="SAME")
    # result_block = result_block[:,(width_pad+1)-1].reshape(-1,1) #reshape promotes to column vec, -1 for 0 indexing
    # print("result_block padded",result_block.shape)

    if norm:
        #norm
        result_block = jax.nn.standardize(result_block, axis=(0,1))

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    return jnp.mean(result_block, axis=-1) # averaged across filters, back to 1 channel dim

@partial(jax.jit, static_argnums=(0,6,7,8,9))
def _forward_full(width, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1)):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion of width.
    Assumes subregion width is centred within universe WRT columns and width is even.

    This version computes the forward across the full array with the nethood applied to it
    The result is a full 2D convolution with the nethood, which can be expensive.
    Multiple nethoods or filters are supported, returns the mean result of these filters

    All operations are done on the entire universe.
    """
    # centre = cells.shape[1]//2
    # padding = filters.shape[1]-1
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape)) # half of all dims of filter
    height_pad_shape = height_pad_shape + (0,) #padding for each dim plus filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    # print("height_pad_shape", height_pad_shape,"height_pad",height_pad)

    # sub_weights = weights[:,centre-width//2-padding:centre+width//2+padding]
    # sub_biases = biases[:,centre-width//2-padding:centre+width//2+padding]
    # sub_states = states[:,centre-width//2-padding:centre+width//2+padding]

    cells = cells[...,jnp.newaxis]
    weights = weights[...,jnp.newaxis]
    biases = biases[...,jnp.newaxis]
    states = states[...,jnp.newaxis]

    #conv to eval neighbourhood
    # padding VALID mode approach, reduced complexity, handles boundaries better?
    result = _fast_conv(cells, filters, dn, window_strides, padding_mode="VALID")
    result = jnp.pad(result, height_pad) #pad rows to undo valid conv in rows
    # result = _fast_conv(cells[:,centre-width//2-padding:centre+width//2+padding], filters, dn)
    # padding SAME mode approach
    # result = _fast_conv(cells, filters, dn, window_strides) #produces result channel dims matching filters channel dims
    # print(result.shape)

    if norm:
        #norm
        result = jax.nn.standardize(result, axis=(0,1))

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    # states = (states - states.min()) / (states.max() - states.min())

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    if activate == None:
        result = result*weights*(1.0-states) + (result+biases)*states
    else:
        result = result*weights*(1.0-states) + activate(result+biases)*states

    return jnp.mean(result, axis=-1) # averaged across filters, back to 1 channel dim

@partial(jax.jit, static_argnums=(0,6,7))
def _forward_fast_1D(index, cells, weights, biases, states, filter_ffts, activate, norm):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    Fast version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index
    """
    width = filter_ffts.shape[1]
    # sub_cells = cells[:,index-1:index-1+width]
    sub_weights = weights[:,index].reshape(-1,1) #reshape promotes to column vec
    sub_biases = biases[:,index].reshape(-1,1) #reshape promotes to column vec
    sub_states = states[:,index].reshape(-1,1) #reshape promotes to column vec
    # print(sub_states.shape)

    #conv
    result_block = _fast_conv_1D(cells[:,index-1:index-1+width], filter_ffts)
    # print(result_block.shape)

    if norm:
        #norm
        result_block = jax.nn.standardize(result_block, axis=(0,1))

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight
    result_block = jnp.where(jnp.logical_and(sub_states>=STATE_SIGNAL,sub_states<STATE_ACTIVE), result_block*sub_weights, result_block) #weight if signal
    # print(result_block.shape)
    if activate == None:
        result_block = jnp.where(sub_states>=STATE_ACTIVE, result_block+sub_biases, result_block) #activate if passive, more accurate
    else:
        result_block = jnp.where(sub_states>=STATE_ACTIVE, activate(result_block+sub_biases), result_block) #activate if passive, more accurate
    # result_block = jnp.where(sub_states==2, activate(result_block+biases), result_block) #activate if passive, faster

    return result_block

@partial(jax.jit, static_argnums=(0,1))
def _fft_1D(N, length, filters):
    """
    2D conv done in 1D convs via the CCP and Fourier space.

    Cells is assumed to be the relevant sub-region to process
    """
    #pad to length of layers
    filters = jnp.pad(filters, ((0,(N-length)), (0,0)), mode='constant')
    filters = jnp.roll(filters, -1, axis=0) #make sure centre pixel is first in filter by periodic shift
    # print("filters:", filters.shape, filters)
    # FFTs of filter
    return jnp.fft.fft(filters, axis=0) #ffts of rows

# conv operators
@partial(jax.jit, static_argnums=(2,3,4,5,6))
def _fast_conv(sub_cells, filters, dn, window_strides, padding_mode="SAME", lhs_dilation=None, rhs_dilation=None):
    """
    2D conv done with filters. Supports multiple filters in channel dim.
    Assumes sub_cells has matching channel dim to filters

    Cells is assumed to be the relevant sub-region to process
    """
    #Conv via 2D brute force
    #get weighted sum of neighbors using convolution
    sub_cells_conv = jax.lax.conv_general_dilated(sub_cells[jnp.newaxis,...], # lhs = NHWC image tensor
                                        filters,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        padding_mode, # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    return sub_cells_conv[0] # NHWC, remove extra batch dim, return multi-filters result

@partial(jax.jit, static_argnums=())
def _fast_conv_1D(sub_cells, filter_ffts):
    """
    2D conv done in 1D convs via the CCP and Fourier space.

    Cells is assumed to be the relevant sub-region to process
    """
    #Conv via CCP
    # FFTs of universe
    cells_conv1D = jnp.fft.fft(sub_cells, axis=0) #ffts of rows
    #convolve in Fourier space
    result_conv1D = filter_ffts * cells_conv1D
    # print("i:", i, sub_universe.shape, sub_universe)
    result_conv1D = jnp.sum(result_conv1D, axis=-1).reshape(-1,1) #reshape promotes to column vec
    # print(result_conv1D.shape, result_conv1D)
    #inverse FFT
    result_block = jnp.fft.ifft(result_conv1D, axis=0) #ffts of rows

    return jnp.real(result_block)

@partial(jax.jit, static_argnums=(2,3,4,5))
def _filter(cells, nethood, dn, window_strides, lhs_dilation=None, rhs_dilation=None):
    """
    Filter the array provided by filter provided.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,...,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        'SAME', # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    return cells[0,...,0] # NHWC

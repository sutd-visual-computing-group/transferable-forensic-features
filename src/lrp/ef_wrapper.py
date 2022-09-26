"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).



from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock, VALID_MODELS

from efficientnet_pytorch.utils import MemoryEfficientSwish, Swish, Conv2dDynamicSamePadding, Conv2dStaticSamePadding

from lrp.ef_lrp_general import *

import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock_canonized(MBConvBlock):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__(block_args, global_params, image_size)

        self.elt = sum_stacked2() 
        self.multdistribute_wtatosecond = mult_wtatosecond()
        
    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1) # dont care as nothing is redistributed here with wta rule
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            #x = torch.sigmoid(x_squeezed) * x # old code
            x = self.multdistribute_wtatosecond.apply( torch.sigmoid(x_squeezed) , x ) # wrap by wta rule

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            #x = x + inputs  # skip connection # old code
            x = self.elt( torch.stack([x,inputs], dim=0) ) 
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


def bnafterconv_overwrite_intoconv_effnet(conv,bn): #after visatt

    
    
        
    print(conv,bn)

    assert (isinstance(bn,nn.BatchNorm2d))
    assert (isinstance(conv,nn.Conv2d))

    s = torch.sqrt(bn.running_var+bn.eps)
    w = bn.weight
    b = bn.bias
    m = bn.running_mean
    conv.weight = torch.nn.Parameter(conv.weight.data * (w / s).reshape(-1, 1, 1, 1))

    #print( 'w/s, conv.bias', (w/s), conv.bias )

    if conv.bias is None:
      conv.bias = torch.nn.Parameter( ((-m) * (w / s) + b).to(conv.weight.dtype) )
    else:
      conv.bias = torch.nn.Parameter(( conv.bias - m) * (w / s) + b)
    #print( ' conv.bias new',  conv.bias )
    return conv



class EfficientNet_canonized(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock_canonized(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock_canonized(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        self.tmpfeats = self._bn0(self._conv_stem(inputs))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        print('pre ex')
        x = self.extract_features(inputs)
        print('post ex')
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    
    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


    def getbyname(self,name):

        def iteratget(obj,components):

          if not hasattr(obj,components[0]):
            return None
          elif len(components)==1:
            z = getattr(obj,components[0])
            #print('found!!', components[0])
            #exit()
            return z
          else:
            nextobj=getattr(obj,components[0])
            return iteratget(nextobj,components[1:])

        components=name.split('.')
        obj=iteratget(self,components)
        return obj
        
    def setbyname(self,name,value):

        def iteratset(obj,components,value):

          if not hasattr(obj,components[0]):
            return False
          elif len(components)==1:
            setattr(obj,components[0],value)
            #print('found!!', components[0])
            #exit()
            return True
          else:
            nextobj=getattr(obj,components[0])
            return iteratset(nextobj,components[1:],value)

        components=name.split('.')
        success=iteratset(self,components,value)
        return success   
                 
    def copyfromefficientnet(self, net, lrp_params, lrp_layer2method):
      
      assert( isinstance(net, EfficientNet))
      if (self._global_params.dropout_rate != 0) or (self._global_params.drop_connect_rate !=0 ):
        print( '(self._global_params.dropout_rate != 0) or (self._global_params.drop_connect_rate !=0 )', self._global_params.dropout_rate, self._global_params.drop_connect_rate ) 
        exit()
      #print(self._global_params) #drop_connect_rate
      #exit()
      #not checked that you create the right model
      

      updated_layers_names=[]

      last_target_module_name=None
      last_target_module=None
      
      for src_module_name, src_module in net.named_modules():
        print('at src_module_name', src_module_name )

        foundsth=False


        if isinstance(src_module, nn.Linear):
          #copy linear layers
          foundsth=True
          print('is Linear')
          #m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
          wrapped = get_lrpwrapperformodule_effnet( copy.deepcopy(src_module) , lrp_params, lrp_layer2method)
          if False== self.setbyname(src_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )
          updated_layers_names.append(src_module_name)
        # end of if


        if isinstance(src_module, (Conv2dDynamicSamePadding, Conv2dStaticSamePadding, nn.Conv2d)):
          #store conv2d layers
          foundsth=True
          print('is Conv2d alike')
          last_src_module_name=src_module_name
          last_src_module=src_module
        # end of if

        if isinstance(src_module, nn.BatchNorm2d):
          # conv-bn chain
          foundsth=True
          print('is BatchNorm2d')

          if (True == lrp_params['use_zbeta']) and (last_src_module_name == '_conv_stem'):
            thisis_inputconv_andiwant_zbeta = True
          else:
            thisis_inputconv_andiwant_zbeta = False

          m = copy.deepcopy(last_src_module)
          m = bnafterconv_overwrite_intoconv_effnet(m , bn = src_module)
          # wrap conv
          wrapped = get_lrpwrapperformodule_effnet( m , lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta = thisis_inputconv_andiwant_zbeta )

          if False== self.setbyname(last_src_module_name, wrapped  ):
            raise Modulenotfounderror("could not find module "+nametofind+ " in target net to copy" )
        
          updated_layers_names.append(last_src_module_name)
          
          # wrap batchnorm  
          wrapped = get_lrpwrapperformodule_effnet( resetbn(src_module) , lrp_params, lrp_layer2method)
          #wrapped = get_lrpwrapperformodule_effnet( copy.deepcopy(src_module) , lrp_params, lrp_layer2method)
          if False== self.setbyname(src_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
          updated_layers_names.append(src_module_name)
        # end of if


        #if False== foundsth:
        #  print('!untreated layer')
        print('\n')
      
      # sum_stacked2 is present only in the targetclass, so must iterate here
      for target_module_name, target_module in self.named_modules():

        if isinstance(target_module, nn.AdaptiveAvgPool2d):
          wrapped = get_lrpwrapperformodule_effnet( target_module , lrp_params, lrp_layer2method)

          if False== self.setbyname(target_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
          updated_layers_names.append(target_module_name)

        if isinstance(target_module, (MemoryEfficientSwish, Swish)):
          wrapped = get_lrpwrapperformodule_effnet( target_module , lrp_params, lrp_layer2method)

          if False== self.setbyname(target_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+src_module_name+ " in target net to copy" )            
          updated_layers_names.append(target_module_name)
          
        # sum_stacked2
        if isinstance(target_module, sum_stacked2 ):
          wrapped =  get_lrpwrapperformodule_effnet( target_module , lrp_params, lrp_layer2method)
          if False== self.setbyname(target_module_name, wrapped ):
            raise Modulenotfounderror("could not find module "+target_module_name+ " in target net , impossible!" )            
          updated_layers_names.append(target_module_name)
          
        # se mult
        if isinstance(target_module, mult_wtatosecond):
          # do nothing is already wrapped
          updated_layers_names.append(target_module_name)

        
      for target_module_name, target_module in self.named_modules():
        if target_module_name not in updated_layers_names:
          print('not updated:', target_module_name)
      #exit()

## for efficientnet as per lukemelas


class Conv2dDynamicSamePadding_zbeta_wrapper_class(nn.Module):
  def __init__(self, module, lrpignorebias,lowest  = None, highest = None  ):
    super(Conv2dDynamicSamePadding_zbeta_wrapper_class, self).__init__()

    if lowest is None:
      lowest=torch.min(torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]))
    if highest is None:
      highest=torch.max (torch.tensor([(1-0.485)/0.229, (1-0.456)/0.224, (1-0.406)/0.225]))
    assert( isinstance( module, nn.Conv2d ))

    self.module=module
    self.wrapper=Conv2dDynamicSamePadding_zbeta_wrapper_fct()

    self.lrpignorebias=lrpignorebias

    self.lowest=lowest
    self.highest=highest

  def forward(self,x):
    y=self.wrapper.apply( x, self.module, self.lrpignorebias, self.lowest, self.highest)
    return y


class Conv2dDynamicSamePadding_zbeta_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias, lowest, highest):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride',  'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)  
              elif isinstance(v, list):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)    
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, lowest.to(module.weight.device), highest.to(module.weight.device), *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d zbeta custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, lowest_, highest_, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride',  'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=False )
        else:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        print('zbeta conv2dconstr weights')

        #pnconv = posnegconv(conv2dclass, ignorebias=True)
        #X = input_.clone().detach().requires_grad_(True)
        #R= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output[0], eps0 = 1e-12, eps=0)

        any_conv =  anysign_conv_efficientnet( module, ignorebias=lrpignorebiastensor.item())

        X = input_.clone().detach().requires_grad_(True)
        L = (lowest_ * torch.ones_like(X)).requires_grad_(True)
        H = (highest_ * torch.ones_like(X)).requires_grad_(True)

        with torch.enable_grad():
            Z = any_conv.forward(mode='justasitis', x=X) - any_conv.forward(mode='pos', x=L) - any_conv.forward(mode='neg', x=H) 
            S = safe_divide(grad_output.clone().detach(), Z.clone().detach(), eps0=1e-6, eps=1e-6)
            Z.backward(S)
            R = (X * X.grad + L * L.grad + H * H.grad).detach()

        print('zbeta conv2d custom R', R.shape )
        #exit()
        return R,None,None,None, None # for  (x, conv2dclass,lrpignorebias, lowest, highest)



#shortcut, eh
class mult_wtatosecond(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x1,x2):
        return x1*x2 # can place in sigmoid here

    @staticmethod
    def backward(ctx,grad_output):
        return torch.zeros_like(grad_output), grad_output


class anysign_conv_efficientnet(nn.Module):


    def _clone_module(self, module):
        clone = Conv2dDynamicSamePadding(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride',  'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super( anysign_conv_efficientnet, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)

      self.jusconv=self._clone_module(conv)
      self.jusconv.weight= torch.nn.Parameter( conv.weight.data.clone() ).to(conv.weight.device)

      if ignorebias==True:
        self.posconv.bias=None
        self.negconv.bias=None
        self.jusconv.bias=None
      else:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) ).to(conv.weight.device)
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) ).to(conv.weight.device)
              self.jusconv.bias= torch.nn.Parameter( conv.bias.data.clone() ).to(conv.weight.device)

      print('done init')

    def forward(self,mode,x):
        if mode == 'pos':
            return self.posconv.forward(x)
        elif mode =='neg':
            return self.negconv.forward(x)
        elif mode =='justasitis':
            return self.jusconv.forward(x)
        else:
            raise NotImplementedError("anysign_conv notimpl mode: "+ str(mode))
        return vp+vn


class posnegconv_efficientnet(nn.Module):


    def _clone_module(self, module):
        clone = Conv2dDynamicSamePadding(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'dilation',  'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super(posnegconv_efficientnet, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)


      #ignbias=True
      #ignorebias=False
      if ignorebias==True:
        self.posconv.bias=None
        self.negconv.bias=None
      else:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) )
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) )

      #print('done init')

    def forward(self,x):
        vp= self.posconv ( torch.clamp(x,min=0)  )
        vn= self.negconv ( torch.clamp(x,max=0)  )
        return vp+vn


class invertedposnegconv_efficientnet(nn.Module):


    def _clone_module(self, module):
        clone = Conv2dDynamicSamePadding(module.in_channels, module.out_channels, module.kernel_size,
                     **{attr: getattr(module, attr) for attr in ['stride', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
      super(invertedposnegconv_efficientnet, self).__init__()

      self.posconv=self._clone_module(conv)
      self.posconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(min=0) ).to(conv.weight.device)

      self.negconv=self._clone_module(conv)
      self.negconv.weight= torch.nn.Parameter( conv.weight.data.clone().clamp(max=0) ).to(conv.weight.device)


      #ignbias=True
      #ignorebias=False

      self.posconv.bias=None
      self.negconv.bias=None
      if ignorebias==False:
          if conv.bias is not None:
              self.posconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(min=0) )
              self.negconv.bias= torch.nn.Parameter( conv.bias.data.clone().clamp(max=0) )

      #print('done init')

    def forward(self,x):
        #vp= self.posconv ( torch.clamp(x,min=0)  )
        #vn= self.negconv ( torch.clamp(x,max=0)  )
        
        vp= self.posconv (  torch.clamp(x,max=0)  ) #on negatives
        vn= self.negconv ( torch.clamp(x,min=0) ) #on positives
        #return vn
        #return vp
        #print( 'negfwd pos?' ,torch.mean((vp>0).float()).item() , torch.mean((vn>0).float()).item() )
        
        return vp+vn # zero or neg




class Conv2dDynamicSamePadding_beta0_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride',  'dilation', 'groups'] #padding omitted
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              
              print(attr,v)
              
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, list):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)     
              else:
                print('v is neither int nor tuple. unexpected', attr,v, type(v))
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'dilation',  'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        #print(paramsdict)
        #print(conv2dbias)
        #exit()

        #print('conv2dconstr')
        if conv2dbias is None:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=False )
        else:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                

        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv_efficientnet(module, ignorebias = lrpignorebiastensor.item())


        #print('conv2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output, eps0 = 1e-12, eps=0)
        #R= lrp_backward(_input= X , layer = pnconv , relevance_output = torch.ones_like(grad_output), eps0 = 1e-12, eps=0)
        #print( 'beta 0 negR' ,torch.mean((R<0).float()).item() ) # no neg relevance

        print('effnet conv2d custom R', R.shape )
        print(module.in_channels, module.out_channels, module.kernel_size, module.stride)
        #exit()
        return R,None, None



class Conv2dDynamicSamePadding_betaany_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias, beta):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)  
              elif isinstance(v, list):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)    
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, beta, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, beta, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride', 'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=False )
        else:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv_efficientnet(module, ignorebias = lrpignorebiastensor.item())
        invertedpnconv = invertedposnegconv_efficientnet(module, ignorebias = lrpignorebiastensor.item())


        #print('conv2d custom input_.shape', input_.shape )

        X = input_.clone().detach().requires_grad_(True)
        R1= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output, eps0 = 1e-12, eps=0)
        R2= lrp_backward(_input= X , layer = invertedpnconv , relevance_output = grad_output, eps0 = -1e-12, eps=0)
        #R2= lrp_backward(_input= X , layer = invertedpnconv , relevance_output = torch.ones_like(grad_output), eps0 = -1e-12, eps=0)
        #print(beta.item(), 'negR, posR' ,torch.mean((R2<0).float()).item(), torch.mean((R2>0).float()).item()  ) #only pos or 0

        R = (1+beta)*R1-beta*R2
        return R, None, None, None




class Conv2dDynamicSamePadding_betaadaptive_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, x, module, lrpignorebias, maxbeta):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            #[module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']}

            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride',  'dilation', 'groups']
            values=[]
            for attr in propertynames:
              v = getattr(module, attr)
              # convert it into tensor
              # has no treatment for booleans yet
              if isinstance(v, int):
                v=  torch.tensor([v], dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, tuple):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device) 
              elif isinstance(v, list):
                ################
                ################
                # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
    
                v= torch.tensor(v, dtype=torch.int32, device= module.weight.device)                     
              else:
                print('v is neither int nor tuple. unexpected')
                exit()
              values.append(v)
            return propertynames,values
        ### end of def classproperties2lists(conv2dclass): #####################


        #stash module config params and trainable params
        propertynames,values=configvalues_totensorlist(module)

        if module.bias is None:
          bias=None
        else:
          bias= module.bias.data.clone()
        lrpignorebiastensor=torch.tensor([lrpignorebias], dtype=torch.bool, device= module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, maxbeta, *values ) # *values unpacks the list

        #print('ctx.needs_input_grad',ctx.needs_input_grad)
        #exit()

        #print('conv2d custom forward')
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        #print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, maxbeta, *values  = ctx.saved_tensors
        #print('retrieved', len(values))
        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values): 
            propertynames=['in_channels', 'out_channels', 'kernel_size', 'stride',  'dilation', 'groups']
            # idea: paramsdict={ n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
            paramsdict={}
            for i,n in enumerate(propertynames):
              v=values[i]
              if v.numel==1:
                  paramsdict[n]=v.item() #to cpu?
              else:
                  alist=v.tolist()
                  #print('alist',alist)
                  if len(alist)==1:
                    paramsdict[n]=alist[0]
                  else:
                    paramsdict[n]= tuple(alist)
            return paramsdict
        #######################################################################
        paramsdict=tensorlist_todict(values)

        if conv2dbias is None:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=False )
        else:
          module=Conv2dDynamicSamePadding( **paramsdict, bias=True )
          module.bias= torch.nn.Parameter(conv2dbias)
                
        #print('conv2dconstr')
        module.weight= torch.nn.Parameter(conv2dweight)

        #print('conv2dconstr weights')

        pnconv = posnegconv_efficientnet(module, ignorebias = lrpignorebiastensor.item())
        invertedpnconv = invertedposnegconv_efficientnet(module, ignorebias = lrpignorebiastensor.item())


        #print('get ratios per output')
        # betatensor =  -neg / conv but care for zeros
        X = input_.clone().detach()
        out = module(X)
        negscores = -invertedpnconv(X)
        betatensor = torch.zeros_like(out)
        #print('out.device',out.device, negscores.device)
        betatensor[ out>0 ] = torch.minimum( negscores[ out>0 ] / out [ out>0 ], maxbeta.to(out.device))
        #betatensor = torch.mean( betatensor ,dim=1, keepdim=True )
        
        
        #print('conv2d custom input_.shape', input_.shape )

        X.requires_grad_(True)
        R1= lrp_backward(_input= X , layer = pnconv , relevance_output = grad_output * (1+betatensor), eps0 = 1e-12, eps=0)
        R2= lrp_backward(_input= X , layer = invertedpnconv , relevance_output = grad_output * betatensor, eps0 = 1e-12, eps=0)

        #print('betatensor',betatensor.shape, R1.shape, out.shape, X.shape)

        R = R1 -R2
        return R, None, None, None





def get_lrpwrapperformodule_effnet(module, lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta=False):

  if isinstance(module, nn.ReLU):
    #return zeroparam_wrapper_class( module , relu_wrapper_fct() )

    key='nn.ReLU'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )

  elif isinstance(module, nn.BatchNorm2d):

    key='nn.BatchNorm2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )

  elif isinstance(module, nn.Linear):

    key='nn.Linear'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default linearlayer_eps_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['linear_eps'] )

  #elif isinstance(module, nn.Conv2d): 
  elif isinstance(module, (Conv2dDynamicSamePadding,Conv2dStaticSamePadding)): 
    if True== thisis_inputconv_andiwant_zbeta:
      #print('unsupported')
      #exit()
      return Conv2dDynamicSamePadding_zbeta_wrapper_class(module , lrp_params['conv2d_ignorebias'])
    else:
      key='nn.Conv2d'
      if key not in lrp_layer2method:
        print("found no dictionary entry in lrp_layer2method for this module name:", key)
        raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

      #default conv2d_beta0_wrapper_fct()
      #autogradfunction = lrp_layer2method[key]()
      #return oneparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )

      autogradfunction = lrp_layer2method[key]()
      

      if type(autogradfunction) == conv2d_beta0_wrapper_fct: # dont want test for derived classes but equality
        return oneparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )
      elif type(autogradfunction) == conv2d_betaany_wrapper_fct: 
        return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_beta']) )
      elif type(autogradfunction) == conv2d_betaadaptive_wrapper_fct: 
        return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_maxbeta']) )
        
      elif type(autogradfunction) == Conv2dDynamicSamePadding_beta0_wrapper_fct: # dont want test for derived classes but equality
        return oneparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )
      elif type(autogradfunction) == Conv2dDynamicSamePadding_betaany_wrapper_fct: 
        return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_beta']) )
      elif type(autogradfunction) == Conv2dDynamicSamePadding_betaadaptive_wrapper_fct: 
        return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_maxbeta']) )         
      else:
        print('unknown autogradfunction', type(autogradfunction) )
        exit()


  elif isinstance(module, nn.AdaptiveAvgPool2d):

    key='nn.AdaptiveAvgPool2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default adaptiveavgpool2d_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['pooling_eps'] )

  elif isinstance(module, nn.AvgPool2d):

    key='nn.AvgPool2d'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default adaptiveavgpool2d_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['pooling_eps'] )
    
    
  elif isinstance(module, nn.MaxPool2d):

      key='nn.MaxPool2d'
      if key not in lrp_layer2method:
        print("found no dictionary entry in lrp_layer2method for this module name:", key)
        raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

      #default maxpool2d_wrapper_fct()
      autogradfunction = lrp_layer2method[key]()
      return zeroparam_wrapper_class( module , autogradfunction = autogradfunction )



  elif isinstance(module, sum_stacked2): # resnet specific

    key='sum_stacked2'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default eltwisesum_stacked2_eps_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['eltwise_eps'] )
  
  elif isinstance(module, clamplayer): # densenet specific

    key='clamplayer'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction = autogradfunction)

  elif isinstance(module, tensorbiased_linearlayer): # densenet specific 
       
    key='tensorbiased_linearlayer'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['linear_eps'] )

  elif isinstance(module, tensorbiased_convlayer): # densenet specific 
       
    key='tensorbiased_convlayer'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    
    if type(autogradfunction) == tensorbiasedconv2d_beta0_wrapper_fct:
      return oneparam_wrapper_class( module , autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'] )
    elif type(autogradfunction) == tensorbiasedconv2d_betaany_wrapper_fct: 
      return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_beta']) )
    elif type(autogradfunction) == tensorbiasedconv2d_betaadaptive_wrapper_fct: 
      return twoparam_wrapper_class(module, autogradfunction = autogradfunction, parameter1 = lrp_params['conv2d_ignorebias'], parameter2 = torch.tensor( lrp_params['conv2d_maxbeta']) )
    else:
      print('unknown autogradfunction', type(autogradfunction) )
      exit()
  elif isinstance(module, (MemoryEfficientSwish, Swish) ):
    #return zeroparam_wrapper_class( module , relu_wrapper_fct() )

    key='Swish'
    if key not in lrp_layer2method:
      print("found no dictionary entry in lrp_layer2method for this module name:", key)
      raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", key)

    #default relu_wrapper_fct()
    autogradfunction = lrp_layer2method[key]()
    return zeroparam_wrapper_class( module , autogradfunction= autogradfunction )
  else:
    print("found no lookup for this module:", module)
    raise lrplookupnotfounderror( "found no lookup for this module:", module)
    



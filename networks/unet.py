import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np

conv_out_shape = lambda W: int(W - 3 + 2) + 1

class UNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=None):
        super().__init__()

        self.activation = activation
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(in_features, 64, kernel_size=3, padding="same") # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding="same") # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding="same") # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding="same") # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding="same") # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding="same") # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding="same") # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding="same") # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding="same") # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding="same") # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding="same")
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding="same")

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding="same")
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding="same")

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding="same")
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding="same")

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding="same")
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding="same")

        # Output layer
        self.outconv = nn.Conv2d(64, out_features, kernel_size=1)

    def forward(self, x):
            # Encoder
            xe11 = relu(self.e11(x))
            xe12 = relu(self.e12(xe11))
            xp1 = self.pool1(xe12)

            xe21 = relu(self.e21(xp1))
            xe22 = relu(self.e22(xe21))
            xp2 = self.pool2(xe22)

            xe31 = relu(self.e31(xp2))
            xe32 = relu(self.e32(xe31))
            xp3 = self.pool3(xe32)

            xe41 = relu(self.e41(xp3))
            xe42 = relu(self.e42(xe41))
            xp4 = self.pool4(xe42)

            xe51 = relu(self.e51(xp4))
            xe52 = relu(self.e52(xe51))
            
            # Decoder
            xu1 = self.upconv1(xe52)
            xu11 = torch.cat([xu1, xe42], dim=1)
            xd11 = relu(self.d11(xu11))
            xd12 = relu(self.d12(xd11))

            xu2 = self.upconv2(xd12)
            xu22 = torch.cat([xu2, xe32], dim=1)
            xd21 = relu(self.d21(xu22))
            xd22 = relu(self.d22(xd21))

            xu3 = self.upconv3(xd22)
            xu33 = torch.cat([xu3, xe22], dim=1)
            xd31 = relu(self.d31(xu33))
            xd32 = relu(self.d32(xd31))

            xu4 = self.upconv4(xd32)
            xu44 = torch.cat([xu4, xe12], dim=1)
            xd41 = relu(self.d41(xu44))
            xd42 = relu(self.d42(xd41))

            # Output layer
            out = self.outconv(xd42)

            return out
    

class EncBlock(nn.Module):
    def __init__(self, in_c, out_c, use_batchnorm=True):
        super().__init__()
        
        self.block = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size=3, padding='same'),
			nn.BatchNorm2d(out_c) if use_batchnorm else nn.Identity(),
			nn.ReLU(),
			nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
			nn.BatchNorm2d(out_c) if use_batchnorm else nn.Identity(),
			nn.ReLU()
		)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        x = self.block(inputs)
        p = self.pool(x)
        return x, p

class DecBlock(nn.Module):
    def __init__(self, in_c, out_c, use_batchnorm=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = EncBlock(out_c+out_c, out_c, use_batchnorm=use_batchnorm).block

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
class AttentionBlockV0(nn.Module):
	def __init__(self, dim, channels, time_features, hidden_dim=64):
		super().__init__()
		self.dim = dim
		self.attention = nn.Sequential(
               nn.Linear(time_features, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, dim*dim),
               nn.Sigmoid()
		)
		self.layerK = nn.Conv2d(channels, channels, kernel_size=1)
		self.layerV = nn.Conv2d(channels, channels, kernel_size=1)
            
	def forward(self, date, x):
		q = self.attention(date).view(-1, 1, self.dim, self.dim)
		k = self.layerK(x)
		v = self.layerV(x)
		# attention map as in an transformer
		att = torch.einsum('bchw,bdhw->bcdhw', q, k)
		att = att / self.dim
		att = torch.softmax(att, dim=1)
		# multiply the attention map by the value
		out = torch.einsum('bcdhw,bdhw->bchw', att, v)
		return out
      
class AttentionBlock(nn.Module):
	def __init__(self, input_dim, output_shape, hidden_dim=64):
		super().__init__()
		assert len(output_shape) == 2, "output_shape should have the form L x W of the desired output, excluding channels"
		self.output_shape = output_shape
		self.fc = nn.Linear(input_dim, output_shape[0] * output_shape[1])
		# TODO: add a second layer?
            
	def forward(self, date, x):
		attention_mask = torch.sigmoid(self.fc(date))
		attention_mask = attention_mask.view(-1, 1, self.output_shape[0], self.output_shape[1])
		return x * attention_mask
          
class ExtraUNet(nn.Module):
	def __init__(self, in_features: int, out_features: int, time_features:int=8, image_shape=(96, 128), activation=None, use_batchnorm=False, use_attention=False):
		super().__init__()

		self.activation = activation
		image_shape = np.array(image_shape)
		
		# Encoder
		# In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
		# Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
		# -------

		# Maxpooling in the last block
		self.encoder = nn.ModuleList([
			EncBlock(in_features, 64, use_batchnorm=use_batchnorm), # image shape after pooling: h/2 x w/2
			EncBlock(64, 128, use_batchnorm=use_batchnorm), # h/4 x w/4
			EncBlock(128, 256, use_batchnorm=use_batchnorm), # h/8 x w/8
			EncBlock(256, 512, use_batchnorm=use_batchnorm) # h/16 x w/16
        ])
        
		self.bottleneck = EncBlock(512, 1024, use_batchnorm=use_batchnorm).block
		
		# Decoder
		self.decoder = nn.ModuleList([
			DecBlock(1024, 512, use_batchnorm=use_batchnorm),
			DecBlock(512, 256, use_batchnorm=use_batchnorm),
			DecBlock(256, 128, use_batchnorm=use_batchnorm),
			DecBlock(128, 64, use_batchnorm=use_batchnorm)
		])

		# Output layer
		self.outconv = nn.Conv2d(64, out_features, kernel_size=1)
        
		self.use_attention = use_attention
		if use_attention:
			self.attention_blocks = nn.ModuleList([
                AttentionBlock(input_dim=time_features, output_shape=image_shape),
                AttentionBlock(input_dim=time_features, output_shape=image_shape//2),
                AttentionBlock(input_dim=time_features, output_shape=image_shape//4),
                AttentionBlock(input_dim=time_features, output_shape=image_shape//8),
            ])
		else:
			self.attention_blocks = [nn.Identity] * 4

	def forward(self, x, date):
		# the "Extra" model has an extra input, the date (a unique number for the whole image), this value is encoded as an attention map and multiplied by the output of all the skip connections
            
		# Encoder
		skips = []
		for block, att_block in zip(self.encoder, self.attention_blocks):
			s, x = block(x)
			if self.use_attention:
				s = att_block(date, s)
			skips.append(s)
                  
		x = self.bottleneck(x)

		# Decoder
		for block, skip in zip(self.decoder, reversed(skips)):
			x = block(x, skip)

		# Output layer
		out = self.outconv(x)

		return out
      
if __name__ == "__main__":
	# Test the model
	model = ExtraUNet(3, 1, use_attention=True)
	print(model)
	x = torch.randn(2, 3, 512, 512)
	date = torch.randn(1, 8)
	out = model(x, date)
	print(out.shape)

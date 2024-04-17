import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np

conv_out_shape = lambda W: int(W - 3 + 2) + 1

class EncBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.):
        super().__init__()
        self.e1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        self.e2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
		
    def forward(self, inputs):
        x = relu(self.e1(inputs))
        x = relu(self.e2(x))
        x = self.dropout(x)
        p = self.pool(x)
        return x, p
	
class DecBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = EncBlock(out_c+out_c, out_c, dropout=dropout)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x, _ = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=None, dropout=0.):
        super().__init__()

        self.activation = activation
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------

        self.enc1 = EncBlock(in_features, 64) # input: 572x572x3
        self.enc2 = EncBlock(64, 128, dropout) # input: 284x284x64
        self.enc3 = EncBlock(128, 256, dropout) # input: 140x140x128
        self.enc4 = EncBlock(256, 512, dropout) # input: 68x68x256
        self.enc5 = EncBlock(512, 1024, dropout) # input: 32x32x512
        
        # Decoder, dropout only here
        self.dec1 = DecBlock(1024, 512, dropout)
        self.dec2 = DecBlock(512, 256, dropout)
        self.dec3 = DecBlock(256, 128, dropout)
        self.dec4 = DecBlock(128, 64, dropout)
        
        # Output layer
        self.outconv = nn.Conv2d(64, out_features, kernel_size=1)
	
    def eval_dp(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):
            # Encoder
            xe12, xp1 = self.enc1(x)
            xe22, xp2 = self.enc2(xp1)
            xe32, xp3 = self.enc3(xp2)
            xe42, xp4 = self.enc4(xp3)
            xe52, _ = self.enc5(xp4)
            
            # Decoder
            xd12 = self.dec1(xe52, xe42)
            xd22 = self.dec2(xd12, xe32)
            xd32 = self.dec3(xd22, xe22)
            xd42 = self.dec4(xd32, xe12)
            
            # Output layer
            out = self.outconv(xd42)
            return out
    
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

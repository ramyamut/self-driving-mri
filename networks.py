import torch
from torch import nn
import torchvision.models as models

import e3cnn
import seq

class UNet(nn.Module):

    def __init__(self,
                 n_input_channels=1,
                 n_output_channels=1,
                 n_levels=3,
                 n_conv=2,
                 n_feat=32,
                 feat_mult=1,
                 kernel_size=3,
                 activation='relu',
                 last_activation=None,
                 batch_norm_after_each_conv=False,
                 residual_blocks=False,
                 encoder_only=False,
                 upsample=False,
                 use_skip_connections=True,
                 rm_top_skip_connection=0,
                 predict_residual=False):

        super(UNet, self).__init__()

        # input/output channels
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

        # general architecture
        self.encoder_only = encoder_only
        self.upsample = upsample
        self.rm_top_skip_connection = rm_top_skip_connection if use_skip_connections else self.n_levels
        self.predict_residual = predict_residual

        # convolution block parameters
        self.n_levels = n_levels
        self.n_conv = n_conv
        self.feat_mult = feat_mult
        self.feat_list = [n_feat * feat_mult ** i for i in range(self.n_levels)]
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm_after_each_conv = batch_norm_after_each_conv
        self.residual_blocks = residual_blocks

        # define convolutional blocks
        self.list_encoder_blocks = self.get_list_encoder_blocks()  # list of length self.n_levels
        if not self.encoder_only:
            self.list_decoder_blocks = self.get_list_decoder_blocks()  # list of length self.n_levels - 1
            self.last_conv = torch.nn.Conv3d(self.feat_list[0], self.n_output_channels, kernel_size=1)
        else:
            self.list_decoder_blocks = []
            self.last_conv = torch.nn.Conv3d(self.feat_list[-1], self.n_output_channels, kernel_size=1)

        if last_activation == 'relu':
            self.last_activation = torch.nn.ReLU()
        elif last_activation == 'elu':
            self.last_activation = torch.nn.ELU()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=1)
        elif last_activation == 'tanh':
            self.last_activation = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        """takes tuple of two inputs, each with the same shape [B, C, H, W, D]"""

        tens = x

        # down-arm
        list_encoders_features = []
        #breakpoint()
        for i, encoder_block in enumerate(self.list_encoder_blocks):
            if i > 0:
                tens = torch.nn.functional.max_pool3d(tens, kernel_size=2)
            tens_out = encoder_block(tens)
            tens = tens + tens_out if self.residual_blocks else tens_out
            list_encoders_features.append(tens)

        # up-arm
        if not self.encoder_only:

            # remove output of last encoder block (i.e. the bottleneck) from the list of features to be concatenated
            list_encoders_features = list_encoders_features[::-1][1:]

            # build conv
            for i in range(len(self.list_decoder_blocks)):
                tens = torch.nn.functional.upsample(tens, scale_factor=2, mode='trilinear')
                if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                    tens_out = torch.cat((list_encoders_features[i], tens), dim=1)
                else:
                    tens_out = tens
                tens_out = self.list_decoder_blocks[i](tens_out)
                tens = tens + tens_out if self.residual_blocks else tens_out

        # final convolution
        tens = self.last_conv(tens)
        if self.last_activation is not None:
            tens = self.last_activation(tens)

        if self.upsample:
            tens = torch.nn.functional.interpolate(tens, scale_factor=2 ** (self.n_levels - 1), mode='trilinear')

        # residual
        if self.predict_residual:
            tens = x + tens

        return tens

    def get_list_encoder_blocks(self):

        list_encoder_blocks = []
        for i in range(self.n_levels):

            # number of input/output feature maps for each convolution
            if i == 0:
                n_input_feat = [self.n_input_channels] + [self.feat_list[i]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[i - 1]] + [self.feat_list[i]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[i]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_encoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_encoder_blocks)

    def get_list_decoder_blocks(self):

        list_decoder_blocks = []
        for i in range(0, self.n_levels - 1):

            # number of input/output feature maps for each convolution
            if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                n_input_feat = [self.feat_list[::-1][i + 1] * (1 + self.feat_mult)] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[::-1][i]] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[::-1][i + 1]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_decoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_decoder_blocks)

    def build_block(self, n_input_feat, n_output_feat):

        # convolutions + activations
        layers = list()
        for conv in range(self.n_conv):
            layers.append(torch.nn.Conv3d(n_input_feat[conv], n_output_feat, kernel_size=self.kernel_size,
                                          padding=self.kernel_size // 2))
            if self.activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif self.activation == 'elu':
                layers.append(torch.nn.ELU())
            else:
                raise ValueError('activation should be relu or elu, had: %s' % self.activation)
            if self.batch_norm_after_each_conv:
                layers.append(torch.nn.BatchNorm3d(n_output_feat))

        # batch norm
        if not self.batch_norm_after_each_conv:
            layers.append(torch.nn.BatchNorm3d(n_output_feat))

        return layers

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self

class E3CNN_Encoder(nn.Module):
    """
    Wrapper class for E3CNN encoder architecture
    """

    def __init__(self, input_chans, output_chans, n_levels, k, last_activation=None, equivariance='O3'):
        super(E3CNN_Encoder, self).__init__()
        
        if last_activation == 'relu':
            self.last_activation = nn.ReLU()
        elif last_activation == 'elu':
            self.last_activation = nn.ELU()
        elif last_activation == 'softmax':
            self.last_activation = nn.Softmax()
        elif last_activation == 'tanh':
            self.last_activation = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None
        self.net = e3cnn.ConvNet(f'{input_chans}x0e',f'{output_chans}x1e + {output_chans*2}x1o',k,k,(1,1,1),n_downsample=n_levels,equivariance=equivariance, lmax=4, pseudo_pool='outer_product', vector_pool='norm_weighted', return_fmaps=True) 
    
    def forward(self, x, uncertainty=False):
        x = self.net.forward(x.to(dtype=torch.float32), pool=False)
        if self.last_activation is not None:
            x[-1] = self.last_activation(x[-1])
        return (x[-1], nn.functional.normalize(x[-2].reshape(3,3,-1), dim=1)) if uncertainty else x[-1]
    
    def pool(self, x):
        return self.net.pool(x)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self

class MovingFrameTransformer(nn.Module):
    """
    Causal transformer: at each timestep t, predicts frame_t from
    measurements_0..t and frames_0..t-1.

    No encoder-decoder split — single causal transformer over T.
    """
    def __init__(self, d_model=128, nhead=4,
                 num_set_layers=2,
                 num_causal_layers=4,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()

        # ── Set pooling (shared across basis directions) ──────────────────────
        self.input_proj = nn.Linear(3, d_model)
        self.set_blocks = nn.ModuleList([
            seq.SetAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_set_layers)
        ])
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Project each basis direction's pooled token
        self.e1_proj = nn.Linear(d_model, d_model)
        self.e2_proj = nn.Linear(d_model, d_model)
        self.e3_proj = nn.Linear(d_model, d_model)

        # Previous frame embedding
        self.frame_proj = nn.Linear(9, d_model)

        # Fuse measurements + previous frame → single token per timestep
        self.token_fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.causal_transformer = seq.RoPETransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_causal_layers, dim_feedforward=dim_feedforward, dropout=dropout)

        self.output_head = nn.Linear(d_model, 9)
    
    def freeze_decoder(self):
        for p in self.causal_transformer.parameters():
            p.requires_grad = False
    
    def freeze_token_fusion(self):
        for p in self.token_fusion.parameters():
            p.requires_grad = False
            
    def freeze_output_head(self):
        for p in self.output_head.parameters():
            p.requires_grad = False
    
    def freeze_frame_proj(self):
        for p in self.frame_proj.parameters():
            p.requires_grad = False

    def pool_set(self, x, pad_mask=None):
        """
        x:    (T, N, 3)   — flattened batch+time
        mask: (T, N)
        →     (T, d_model)
        """
        if isinstance(x, list):
            x = [self.input_proj(elem) for elem in x]
            for block in self.set_blocks:
                x = [block(elem) for elem in x]
            pooled = torch.stack([self.pool_attn(self.pool_query, elem.unsqueeze(0), elem.unsqueeze(0))[0].squeeze() for elem in x])
        else:
            x = self.input_proj(x)
            for block in self.set_blocks:
                x = block(x, pad_mask)
            q = self.pool_query.expand(x.size(0), 1, -1)
            pooled = self.pool_attn(q, x, x, pad_mask)[0].squeeze(1)
        return pooled  # (B*T, d_model)

    def make_causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, batch):
        """
        e1:             (B, T, 2N, 3)
        e2, e3:         (B, T,  N, 3)
        frames_shifted: (B, T,  9)    — target frames shifted right by 1
                                        (start token + frames[:-1])
        mask_e1:        (B, T, 2N)
        mask_e2/e3:      (B, T,  N)
        temp_mask:      (B, T)        — True where T is padded
        """
        # B, T = e1.shape[:2]
        B = len(batch)

        p1 = [self.e1_proj(self.pool_set(b['x'], None if b['vec_pad'] is None else torch.cat([b['vec_pad'], b['vec_pad']], dim=1))) for b in batch]  # (B, T, d_model)

        p2 = [self.e2_proj(self.pool_set(b['y'], b['vec_pad'])) for b in batch]
        p3 = [self.e3_proj(self.pool_set(b['z'], b['vec_pad'])) for b in batch]

        # ── 2. Embed previous frames ──────────────────────────────────────────
        rot = [b['rot'] for b in batch]
        times = [b['t'] for b in batch]
        if self.training:
            rot_output = [seq.shift_frame_sequence(r) for r in rot]
        else:
            rot_output = [r.permute(0,2,1).reshape(-1,9) for r in rot]
        pf = [self.frame_proj(r) for r in rot_output]   # (B, T, d_model)

        # ── 3. Fuse into one token per timestep ───────────────────────────────
        tokens = [self.token_fusion(torch.cat(p,dim=-1)) for p in zip(p1,p2,p3,pf)] # [T,32]
        tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        times_padded = torch.nn.utils.rnn.pad_sequence(times, batch_first=True)

        temp_mask = ~torch.any(tokens_padded,dim=-1)

        # ── 4. Causal self-attention over T ───────────────────────────────────
        T = tokens_padded.shape[1]
        causal_mask = self.make_causal_mask(T, tokens_padded.device)
        out = self.causal_transformer(
            tokens_padded,
            times_padded,
            attn_mask=causal_mask,
            key_padding_mask=temp_mask,
        )

        raw = self.output_head(out).reshape(B, T, 3, 3)            # (B, T, 9)
        normalized = torch.nn.functional.normalize(raw, dim=-1)
        before_proj = normalized.permute(0,1,3,2)
        pred_rot = seq.project_to_SO3(before_proj)
        return before_proj, pred_rot, temp_mask
    
    @torch.no_grad()
    def inference_autoregressive(self, x, y, z, times):
        """Autoregressively predict one frame at a time."""
        self.eval()
        T = len(x)

        import time

        prev_frames = torch.eye(3, device=x[0].device).unsqueeze(0)  # start token
        predictions = []

        start_total = time.time()

        for t in range(T):

            out = self.forward([{'x': x[:t+1], 'y': y[:t+1], 'z': z[:t+1], 'rot': prev_frames, 't': times[:t+1], 'vec_pad': None}])  # (B, T, 3, 3)
 
            next_frame = out[1][0, t].unsqueeze(0)                            # (B, 3, 3) — prediction at t
            predictions.append(next_frame)

            prev_frames = torch.cat([prev_frames, next_frame], dim=0)
        final_pred = torch.stack(predictions, dim=1)  # (B, T, 3, 3)

        if final_pred.shape[1] > 1:
            print(final_pred[0,1])

        return final_pred
    
    @torch.no_grad()
    def inference(self, measurements):
        """Autoregressively predict one frame at a time."""
        self.eval()
        T = len(measurements['t'])
        measurements['vec_pad'] = None
        if measurements['rot'] is None:
            measurements['rot'] = torch.eye(3).unsqueeze(0).to(measurements['x'][0].device).float()

        out = self.forward([measurements])  # (B, T, 3, 3)

        next_frame = out[1][0, T-1].unsqueeze(0) # (B, 3, 3) — prediction at t

        measurements['rot'] = torch.cat([measurements['rot'], next_frame], dim=0)

class IQACNN(nn.Module):

    def __init__(self, model_name = 'resnet18', include_weights: bool = True, in_channels = 3):
        super().__init__()
        
        num_classes = 2

        # Function for Using Weights
        get_weights = lambda m : m if include_weights else None

        # Download ResNet-18 & Update Final Layer
        if model_name == 'resnet18':
            self.model = models.resnet18(weights = get_weights(models.ResNet18_Weights.IMAGENET1K_V1))
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights = get_weights(models.ResNet34_Weights.IMAGENET1K_V1))
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights = get_weights(models.ResNet50_Weights.IMAGENET1K_V1))
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights = get_weights(models.ResNet101_Weights.IMAGENET1K_V1))
        elif model_name == 'resnet152':
            self.model = models.resnet152(weights = get_weights(models.ResNet152_Weights.IMAGENET1K_V1))
        elif model_name == 'convnext_tiny':
            self.model = models.convnext_tiny(weights = get_weights(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
        
        # Update First Layer for 2-channel input for ResNet

        if in_channels == 2 and 'resnet' in model_name:
            # Save pretrained weights
            pretrained_w = self.model.conv1.weight  # (64, 3, 7, 7)

            # Replace conv1 to accept in_channels (2)
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Initialize new conv weights
            if include_weights:
                with torch.no_grad():
                    self.model.conv1.weight[:, 0:1, :, :] = pretrained_w[:, 0:1, :, :]       # actual image
                    self.model.conv1.weight[:, 1:2, :, :] = torch.randn(64, 1, 7, 7) * 0.01  # mask channel


        # Update Final Layer for Binary Classification
        if 'resnet' in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 2) # binary classification
        elif 'convnext' in model_name:
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
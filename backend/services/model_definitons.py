import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MAX_LENGTH      = 35
EMBED_DIM       = 256
LSTM_UNITS      = 512
CNN_FEAT_DIM    = 4096
V2_FEAT_DIM     = 512
DROPOUT         = 0.4
BEAM_WIDTH      = 3

class CNNBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → [MaxPool2d]"""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

class CNNEncoder(nn.Module):
    """
    5-block CNN that outputs a (B, CNN_FEAT_DIM) feature vector per image.
    The feature vector becomes the initial hidden state of the LSTM decoder.
    """
    def __init__(self, feat_dim=CNN_FEAT_DIM, dropout=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(3,   32),            # 224 → 112
            CNNBlock(32,  64),            # 112 → 56
            CNNBlock(64,  128),           # 56  → 28
            CNNBlock(128, 256),           # 28  → 14
            CNNBlock(256, 512, pool=False),  # 14 → 14 (no pool)
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))  # (B, 512, 14, 14) → (B, 512, 1, 1)
        self.flatten = nn.Flatten()                   # → (B, 512)
        self.proj    = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)   # (B, 512, 14, 14)
        x = self.pool(x)        # (B, 512,  1,  1)
        x = self.flatten(x)     # (B, 512)
        x = self.proj(x)        # (B, feat_dim)
        return x


class LSTMDecoder(nn.Module):
    """
    2-layer LSTM that generates captions word-by-word.
    Image features initialise the hidden state so every generation
    step is conditioned on the image (fixes the original add() bug).
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 lstm_units=LSTM_UNITS, feat_dim=CNN_FEAT_DIM,
                 max_length=MAX_LENGTH, dropout=DROPOUT):
        super().__init__()
        self.max_length = max_length
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop       = nn.Dropout(dropout)
        self.img_proj   = nn.Linear(feat_dim, lstm_units)   # image → h0
        self.lstm       = nn.LSTM(embed_dim, lstm_units,
                                  num_layers=2, batch_first=True,
                                  dropout=dropout)
        self.fc         = nn.Linear(lstm_units, vocab_size)

    def forward(self, features, captions):
        """
        Training forward pass (teacher forcing).
        features : (B, feat_dim)   — from CNN
        captions : (B, MAX_LENGTH) — [<start>, w1, ..., wN, <end>, <pad>...]
        Returns logits : (B, MAX_LENGTH-1, vocab_size)
        Target is captions[:, 1:]  i.e. [w1, ..., wN, <end>, <pad>...]
        """
        # Image feature → initial hidden + cell state
        h0 = self.img_proj(features).unsqueeze(0).repeat(2, 1, 1)  # (2, B, lstm_units)
        c0 = torch.zeros_like(h0)

        # Embed all tokens except the last (we never feed <end> as input)
        embeds = self.drop(self.embedding(captions[:, :-1]))  # (B, MAX_LENGTH-1, embed_dim)

        out, _ = self.lstm(embeds, (h0, c0))   # (B, MAX_LENGTH-1, lstm_units)
        logits = self.fc(out)                   # (B, MAX_LENGTH-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate_greedy(self, feature, vocab):
        """
        Greedy decoding — always pick the highest-probability word.

        FIX for original cell 37: vocab.idx2word is a plain dict, so
        each lookup is O(1). The original looped over the entire
        tokenizer.word_index (~8000 entries) for every single word generated.
        """
        self.eval()
        h = self.img_proj(feature).unsqueeze(0).repeat(2, 1, 1)
        c = torch.zeros_like(h)

        word_idx = vocab.word2idx['<start>']
        result   = []

        for _ in range(self.max_length):
            token  = torch.tensor([[word_idx]], device=feature.device)
            embed  = self.drop(self.embedding(token))      # (1, 1, embed_dim)
            out, (h, c) = self.lstm(embed, (h, c))         # (1, 1, lstm_units)
            word_idx = self.fc(out.squeeze(1)).argmax(-1).item()  # O(1)
            word     = vocab.idx2word.get(word_idx, '<unk>')      # O(1)
            if word == '<end>':
                break
            if word not in ('<pad>', '<start>'):
                result.append(word)

        return ' '.join(result)

    @torch.no_grad()
    def generate_beam(self, feature, vocab, beam_width=BEAM_WIDTH):
        """
        Beam search — keep top-k candidate sequences alive at every step.
        Produces noticeably better captions than greedy at low extra cost.
        """
        self.eval()
        h0 = self.img_proj(feature).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        # Each beam: (cumulative_log_prob, token_list, h, c)
        beams     = [(0.0, [start_idx], h0, c0)]
        completed = []

        for _ in range(self.max_length):
            if not beams:
                break
            new_beams = []
            for score, tokens, h, c in beams:
                if tokens[-1] == end_idx:
                    completed.append((score, tokens))
                    continue
                last  = torch.tensor([[tokens[-1]]], device=feature.device)
                embed = self.drop(self.embedding(last))         # (1, 1, embed_dim)
                out, (h_new, c_new) = self.lstm(embed, (h, c))
                log_p = F.log_softmax(self.fc(out.squeeze(1)), dim=-1)  # (1, vocab)
                top_vals, top_ids = log_p.topk(beam_width)
                for i in range(beam_width):
                    new_beams.append((
                        score + top_vals[0, i].item(),
                        tokens + [top_ids[0, i].item()],
                        h_new, c_new
                    ))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        completed.extend(beams)
        best = max(completed, key=lambda x: x[0])
        return vocab.decode(best[1])


class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)              # (B, CNN_FEAT_DIM)
        logits   = self.decoder(features, captions)  # (B, MAX_LENGTH-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, image, vocab, method='beam'):
        """image: (1, 3, H, W) tensor on DEVICE"""
        feature = self.encoder(image)
        if method == 'beam':
            return self.decoder.generate_beam(feature, vocab)
        return self.decoder.generate_greedy(feature, vocab)
    





class PretrainedEncoder(nn.Module):
    """
    ResNet50 with the final avgpool + fc removed.
    Outputs a spatial feature map (B, 49, V2_FEAT_DIM) instead of a single vector.
    49 = 7x7 spatial grid — each cell represents one region of the image.
    """
    def __init__(self, feat_dim=V2_FEAT_DIM, dropout=DROPOUT):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Drop last 2 layers (AdaptiveAvgPool + Linear classifier)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, 7, 7)

        # Project 2048 → feat_dim to reduce memory and computation
        self.proj = nn.Linear(2048, feat_dim)
        self.drop = nn.Dropout(dropout)

        # Freeze all layers except layer4 (last residual block)
        # Early layers detect edges/textures — already perfect, don't touch
        for name, param in self.resnet.named_parameters():
            param.requires_grad = 'layer4' in name

    def forward(self, x):
        with torch.set_grad_enabled(self.training):
            feat = self.resnet(x)                      # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, 49, 2048)
        feat = self.drop(self.proj(feat))              # (B, 49, feat_dim)
        return feat


class BahdanauAttention(nn.Module):
    """
    Soft attention over spatial image features.
    Returns context vector (weighted sum of regions) and alpha (attention map).
    """
    def __init__(self, feat_dim, lstm_units, attn_dim=256):
        super().__init__()
        self.W_feat   = nn.Linear(feat_dim,   attn_dim)
        self.W_hidden = nn.Linear(lstm_units, attn_dim)
        self.v        = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, features, hidden):
        # features : (B, 49, feat_dim)
        # hidden   : (B, lstm_units)
        att_feat   = self.W_feat(features)                 # (B, 49, attn_dim)
        att_hidden = self.W_hidden(hidden).unsqueeze(1)    # (B,  1, attn_dim)
        energy     = torch.tanh(att_feat + att_hidden)     # (B, 49, attn_dim)
        alpha      = torch.softmax(self.v(energy), dim=1)  # (B, 49, 1)
        context    = (alpha * features).sum(dim=1)         # (B, feat_dim)
        return context, alpha.squeeze(-1)                  # (B, feat_dim), (B, 49)


class AttentionLSTMDecoder(nn.Module):
    """
    LSTM decoder that re-attends to the image at every word generation step.

    V1 difference: V1 fed image as h0 once. V2 recomputes image context
    at every timestep using attention — the image is consulted 35 times,
    once per word position, with a different focus each time.

    Also adds length normalisation in beam search to prevent short caption bias.
    """
    def __init__(self, vocab_size, feat_dim=V2_FEAT_DIM,
                 embed_dim=EMBED_DIM, lstm_units=LSTM_UNITS,
                 max_length=MAX_LENGTH, dropout=DROPOUT):
        super().__init__()
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(feat_dim, lstm_units)
        self.drop      = nn.Dropout(dropout)

        # Initialise h, c from mean-pooled spatial features
        self.init_h = nn.Linear(feat_dim, lstm_units)
        self.init_c = nn.Linear(feat_dim, lstm_units)

        # LSTMCell: input = [embedding || context]
        self.lstm = nn.LSTMCell(embed_dim + feat_dim, lstm_units)
        self.fc   = nn.Linear(lstm_units, vocab_size)

    def _init_hidden(self, features):
        mean = features.mean(dim=1)                  # (B, feat_dim)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def forward(self, features, captions):
        """Teacher-forcing: step through each token position."""
        h, c    = self._init_hidden(features)
        logits  = []
        seq_len = captions.size(1) - 1              # exclude last token

        for t in range(seq_len):
            context, _  = self.attention(features, h)
            embed       = self.drop(self.embedding(captions[:, t]))
            lstm_in     = torch.cat([embed, context], dim=1)
            h, c        = self.lstm(lstm_in, (h, c))
            logits.append(self.fc(self.drop(h)))

        return torch.stack(logits, dim=1)            # (B, seq_len, vocab_size)

    @torch.no_grad()
    def generate_beam(self, features, vocab, beam_width=BEAM_WIDTH):
        """Beam search with length normalisation — fixes short caption bias."""
        self.eval()
        h, c      = self._init_hidden(features)
        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        beams     = [(0.0, [start_idx], h, c)]
        completed = []

        for _ in range(self.max_length):
            if not beams:
                break
            new_beams = []
            for score, tokens, h, c in beams:
                if tokens[-1] == end_idx:
                    completed.append((score, tokens))
                    continue
                last    = torch.tensor([tokens[-1]], device=features.device)
                embed   = self.drop(self.embedding(last))
                context, _ = self.attention(features, h)
                lstm_in = torch.cat([embed, context], dim=1)
                h_new, c_new = self.lstm(lstm_in, (h, c))
                log_p   = F.log_softmax(self.fc(h_new), dim=-1)
                top_vals, top_ids = log_p.topk(beam_width)
                for i in range(beam_width):
                    new_beams.append((
                        score + top_vals[0, i].item(),
                        tokens + [top_ids[0, i].item()],
                        h_new, c_new
                    ))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        completed.extend(beams)
        # Length normalisation: divide by sequence length to avoid short bias
        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return vocab.decode(best[1])
    

class ImageCaptionerV2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = PretrainedEncoder(feat_dim=V2_FEAT_DIM)
        self.decoder = AttentionLSTMDecoder(vocab_size, feat_dim=V2_FEAT_DIM)

    def forward(self, images, captions):
        features = self.encoder(images)              # (B, 49, V2_FEAT_DIM)
        logits   = self.decoder(features, captions)  # (B, seq_len-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, image, vocab, method='beam'):
        features = self.encoder(image)
        return self.decoder.generate_beam(features, vocab)
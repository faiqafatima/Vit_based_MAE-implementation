import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

#Core Model Classes

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6, num_patches=196):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, 16 * 16 * 3)
    def forward(self, x, ids_restore):
        B, L, D = x.shape
        mask_tokens = self.mask_token.repeat(B, 196 - L, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return self.pred(self.norm(x))

class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(768, 768)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 768), requires_grad=False)
        self.encoder = MAEEncoder()
        self.enc_to_dec = nn.Linear(768, 384)
        self.decoder = MAEDecoder()

    def patchify(self, imgs):
        p = 16
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

    def unpatchify(self, x):
        p = 16
        h = w = int(x.shape[1]**.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], 3, h * p, h * p))

    def forward(self, imgs, mask_ratio=0.75):
        x = self.patch_embed(self.patchify(imgs)) + self.pos_embed
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_vis = torch.gather(x, 1, ids_shuffle[:, :len_keep].unsqueeze(-1).expand(-1, -1, D))
        latent = self.enc_to_dec(self.encoder(x_vis))
        pred = self.decoder(latent, ids_restore)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return pred, mask



st.set_page_config(page_title="VisionRevive", layout="wide")


st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1 { color: #1e3a8a; font-weight: 800; }
    .result-container { border: 1px solid #e5e7eb; border-radius: 15px; padding: 20px; margin-bottom: 25px; background: #f9fafb; }
    </style>
    """, unsafe_allow_html=True)

st.title("✨ VisionRevive: Insect AI")
st.write("Upload insect images to see the Masked Autoencoder reconstruct them from partial data.")

# Initialize Session State to store up to 5 results
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_model():
    model = MaskedAutoencoder()
    try:
        state_dict = torch.load("best_model.pth", map_location='cpu')
        model.load_state_dict(state_dict)
        return model.eval(), True
    except:
        return model.eval(), False

model, is_loaded = load_model()

# Sidebar controls
with st.sidebar:
    st.header("Control Panel")
    if not is_loaded:
        st.error("Weights not found! Place 'best_model.pth' in the folder.")
    mask_val = st.slider("Masking Intensity", 0.1, 0.9, 0.75)
    uploaded_file = st.file_uploader("Upload New Image", type=['jpg', 'jpeg', 'png'])
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Processing Logic
if uploaded_file:
    
    file_id = uploaded_file.name + str(mask_val)
    
    if not any(item['id'] == file_id for item in st.session_state.history):
        img = Image.open(uploaded_file).convert('RGB')
        
        # Prepare Tensor
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        inp = t(img).unsqueeze(0)
        
        with torch.no_grad():
            pred, mask = model(inp, mask_ratio=mask_val)
        
        # Denorm Logic
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        def denorm(t): return torch.clamp(t * std + mean, 0, 1).permute(1, 2, 0).numpy()

        orig_np = denorm(inp.squeeze())
        recon_np = denorm(model.unpatchify(pred).squeeze())
        m_vis = mask.reshape(14, 14).repeat_interleave(16, 0).repeat_interleave(16, 1)
        masked_np = denorm(inp.squeeze() * (1 - m_vis))

        # Add to history
        st.session_state.history.insert(0, {
            'id': file_id,
            'gt': orig_np,
            'masked': masked_np,
            'recon': recon_np,
            'ratio': mask_val
        })
        if len(st.session_state.history) > 5:
            st.session_state.history.pop()

# Display Results
if not st.session_state.history:
    st.info("Upload an image in the sidebar to start testing.")
else:
    for i, res in enumerate(st.session_state.history):
        with st.container():
            st.markdown(f"<div class='result-container'>", unsafe_allow_html=True)
            st.markdown(f"### Test Result {len(st.session_state.history) - i} (Masked at {int(res['ratio']*100)}%)")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(res['gt'], caption="Ground Truth", use_container_width=True)
            with c2:
                st.image(res['masked'], caption="Masked Input", use_container_width=True)
            with c3:
                st.image(res['recon'], caption="MAE Reconstruction", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
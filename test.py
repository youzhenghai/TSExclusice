import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoProcessor, WhisperModel,
    Wav2Vec2FeatureExtractor, WavLMModel
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── 0. 本地模型目录 ───
WHISPER_DIR = "/home/youzhenghai/model/whisper-large-v3"
WAVLM_DIR   = "/home/youzhenghai/model/wavlm-large"

# ─── 1. 加载并冻结预训练模型主体 ───
processor = AutoProcessor.from_pretrained(
    WHISPER_DIR, local_files_only=True, cache_dir=WHISPER_DIR
)
whisper = WhisperModel.from_pretrained(
    WHISPER_DIR, local_files_only=True, cache_dir=WHISPER_DIR
)
wavlm_fe = Wav2Vec2FeatureExtractor.from_pretrained(
    WAVLM_DIR, local_files_only=True, cache_dir=WAVLM_DIR
)
wavlm = WavLMModel.from_pretrained(
    WAVLM_DIR, local_files_only=True, cache_dir=WAVLM_DIR
)

# 冻结主体 + eval 模式
for p in whisper.parameters(): p.requires_grad = False
for p in wavlm.parameters():   p.requires_grad = False
whisper.eval()
wavlm.eval()

# ─── 2. Batch 特征提取函数 ───
def extract_whisper_features_batch(waves: torch.Tensor, sr: int = 16000):
    """
    waves: Tensor of shape [B, T]
    returns: Tensor of shape [B, T_w, D_w]
    """
    # 将每条 wave 转成 numpy 一维数组
    wav_np_list = [w.cpu().numpy().astype(np.float32) for w in waves]  # len=B

    # 用底层 feature_extractor 处理（只负责音频，不触发 tokenizer 逻辑）
    fe = processor.feature_extractor(
        wav_np_list,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,               # 批量 pad
        return_attention_mask=False
    )
    # fe.input_features: [B, D_w, T_w]
    print("whisper extract:", fe.input_features.shape)
    # 转成 [B, T_w, D_w]
    return fe.input_features.permute(0, 2, 1).to(waves.device)

def extract_wavlm_features_batch(waves: torch.Tensor, sr: int = 16000):
    """
    waves: Tensor of shape [B, T]
    returns: Tensor of shape [B, T_l, D_l]
    """
    wav_np_list = [w.cpu().numpy().astype(np.float32) for w in waves]
    fe = wavlm_fe(
        wav_np_list,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    vals = fe.input_values.to(waves.device)      # [B, T_pad]
    hidden = wavlm(vals).last_hidden_state       # [B, T_pad, D_l]
    print("wavlm extract:", hidden.shape)
    return hidden

# ─── 3. 双路 LoRA 注入 ───
lora_whisper = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
lora_whisper.base_model_name_or_path = WHISPER_DIR
whisper = get_peft_model(whisper, lora_whisper)

lora_wavlm = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
lora_wavlm.base_model_name_or_path = WAVLM_DIR
wavlm = get_peft_model(wavlm, lora_wavlm)

# ─── 4. 下游模块定义 ───
class SpeakerEncoder(nn.Module):
    def __init__(self, in_dim=16000, emb_dim=64):
        super().__init__()
        self.lin = nn.Linear(in_dim, emb_dim)
    def forward(self, x): return self.lin(x)

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, dim)
    def forward(self, x): return self.up(self.act(self.down(x)))

class DualEncoder(nn.Module):
    def __init__(self, dw, dl, df=512):
        super().__init__()
        self.ad_w = Adapter(dw)
        self.ad_l = Adapter(dl)
        self.proj = nn.Linear(dw+dl, df)
    def forward(self, hw, hl):
        return self.proj(torch.cat([self.ad_w(hw), self.ad_l(hl)], dim=-1))

class FlowSynth(nn.Module):
    def __init__(self, df=512, ds=64, mel_bins=80):
        super().__init__()
        self.lin = nn.Linear(df+ds, mel_bins)
    def forward(self, H, e):
        B,T,_ = H.shape
        e_exp = e.unsqueeze(1).expand(-1,T,-1)
        return self.lin(torch.cat([H,e_exp], dim=-1))

class Decoder(nn.Module):
    def __init__(self, df=512, vocab_size=1000):
        super().__init__()
        self.lin = nn.Linear(df, vocab_size)
    def forward(self, H):
        return self.lin(H).permute(0,2,1)

class TSEModel(nn.Module):
    def __init__(self, dw, dl):
        super().__init__()
        self.spk_enc = SpeakerEncoder()
        self.dual    = DualEncoder(dw, dl, df=512)
        self.flow    = FlowSynth(df=512, ds=64)
        self.decoder = Decoder(df=512, vocab_size=1000)
    def forward(self, hf, lf, enroll, txt):
        e = self.spk_enc(enroll)                  # [B, 64]
        H = self.dual(hf, lf)                     # [B, T, 512]
        mel = self.flow(H, e)                     # [B, T, 80]
        logits = self.decoder(H)                  # [B, V, T]
        mel_gt = torch.randn_like(mel)
        l_flow = ((mel-mel_gt)**2).mean()
        l_ce   = nn.CrossEntropyLoss()(logits, txt)
        return l_flow, l_ce

# # ─── 5. 测试主流程 ───
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 特征维度
#     #Dw = processor.feature_extractor.config["n_mels"]  # e.g. 80
#     Dw = 80
#     Dl = wavlm.config.hidden_size                     # e.g. 1024

#     model = TSEModel(dw=Dw, dl=Dl).to(device)
#     model.train()

#     # 随机波形 batch
#     bs, L = 2, 16000
#     mix_wave    = torch.randn(bs, L).to(device)
#     enroll_wave = torch.randn(bs, L).to(device)
#     T_dec       = 10
#     transcript  = torch.randint(0, 1000, (bs, T_dec)).to(device)

#     # —— 批量特征提取 —— 
#     whisper_feats = extract_whisper_features_batch(mix_wave)  # [B, T_w, Dw]
#     wavlm_feats   = extract_wavlm_features_batch(mix_wave)    # [B, T_l, Dl]

#     # —— 对齐长度 —— 
#     T_min = min(whisper_feats.shape[1], wavlm_feats.shape[1])
#     whisper_feats = whisper_feats[:, :T_min, :]
#     wavlm_feats   = wavlm_feats[:, :T_min, :]

#     # —— 前向 & Loss —— 
#     lf, lc = model(whisper_feats, wavlm_feats, enroll_wave, transcript)
#     print(f"loss_flow={lf.item():.4f}, loss_ce={lc.item():.4f}")
# ─── 2. 测试主流程 ───
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 维度
    #Dw=processor.feature_extractor.config["n_mels"]  # e.g. 80
    Dw = 80
    Dl=wavlm.config.hidden_size                     # e.g. 1024

    model=TSEModel(Dw,Dl).to(device); model.train()

    bs,L=2,16000
    mix=torch.randn(bs,L).to(device)
    enroll=torch.randn(bs,L).to(device)
    T_dec=10
    txt=torch.randint(0,1000,(bs,T_dec)).to(device)

    # ** 用随机特征 ** 模拟
    # 你可以把下两行换成真正的 extract_whisper... 逻辑
    # 只要保证形状对上就行
    T= mix.shape[1]//200  # 随便定个 factor
    hf=torch.randn(bs,T,Dw).to(device)
    lf=torch.randn(bs,T,Dl).to(device)

    l1,l2=model(hf,lf,enroll,txt)
    print("loss_flow=",l1.item(), "loss_ce=",l2.item())
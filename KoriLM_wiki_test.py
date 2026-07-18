import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# =====================
# 설정
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLOCK_SIZE = 256
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512

MODEL_FILE = "checkpoint.pt"
TOKENIZER_FILE = "kori_tokenizer.model"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 50

# =====================
# Tokenizer
# =====================
sp = spm.SentencePieceProcessor()
if not sp.load(TOKENIZER_FILE):
    raise Exception("토크나이저 파일 없음")

# [안전 장치] 하드코딩된 크기 대신 토크나이저의 실제 어휘 사전 크기를 모델에 적용합니다.
VOCAB_SIZE = sp.get_piece_size()

# =====================
# Model
# =====================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            N_EMBD,
            N_HEAD,
            batch_first=True
        )
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            )
        )

    def forward(self, x):
        size = x.size(1)
        mask = self.mask[:size, :size]
        out, _ = self.attn(
            x, x, x,
            attn_mask=~mask.bool()
        )
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, N_EMBD * 4),
            nn.GELU(),
            nn.Linear(N_EMBD * 4, N_EMBD)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class KoriLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            N_EMBD
        )
        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )
        self.blocks = nn.Sequential(
            *[Block() for _ in range(N_LAYER)]
        )
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(pos)
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

# =====================
# Load Model
# =====================
def load_model():
    model = KoriLM().to(DEVICE)
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    
    # 만약 체크포인트의 임베딩 레이어 크기가 현재 토크나이저와 다를 경우 예외 처리
    state_dict = checkpoint["model"]
    checkpoint_vocab = state_dict["token_embedding.weight"].shape[0]
    if checkpoint_vocab != VOCAB_SIZE:
        print(f"[알림] 체크포인트의 Vocab 크기({checkpoint_vocab})에 맞춰 가중치를 로드합니다.")
        model.token_embedding = nn.Embedding(checkpoint_vocab, N_EMBD).to(DEVICE)
        model.head = nn.Linear(N_EMBD, checkpoint_vocab).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    print("==============================")
    print("      KoriLM Chat")
    print("==============================")
    print("Device:", DEVICE)
    print("Training step:", checkpoint.get("step", "unknown"))
    print("==============================")
    print("종료 : exit")
    print()

    return model

# =====================
# Generate
# =====================
@torch.no_grad()
def generate(model, text):
    tokens = sp.encode(text)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(MAX_NEW_TOKENS):
        x_crop = x[:, -BLOCK_SIZE:]
        logits = model(x_crop)
        logits = logits[:, -1, :]
        logits /= TEMPERATURE

        # -------------------------------------------------------
        # 수정된 Top-K 필터링 및 샘플링 로직
        # -------------------------------------------------------
        # 1. 차원 축을 기준으로 상위 K개의 값 추출
        v, _ = torch.topk(logits, TOP_K, dim=-1)
        
        # 2. 상위 K번째 값보다 작은 나머지 모든 Logits을 음의 무한대(-inf)로 대체
        logits[logits < v[:, -1:]] = float('-inf')
        
        # 3. 소프트맥스를 취하면 -inf를 가졌던 하위 토큰들은 확률이 정확히 0이 됩니다.
        probs = F.softmax(logits, dim=-1)
        
        # 4. 안전하게 다항 분포 샘플링을 수행합니다.
        next_token = torch.multinomial(probs, num_samples=1)
        # -------------------------------------------------------

        x = torch.cat([x, next_token], dim=1)

        # EOS 종료
        if next_token.item() == sp.eos_id():
            break

    # 최상위 배치([0])의 토큰 리스트를 원래 문장으로 디코딩
    result = sp.decode(x[0].tolist())
    return result

# =====================
# Chat
# =====================
def chat():
    model = load_model()

    while True:
        try:
            q = input("\n너 : ")
        except KeyboardInterrupt:
            print("\n종료")
            break

        if q.strip() == "":
            continue

        if q.lower() == "exit":
            print("KoriLM 종료")
            break

        answer = generate(model, q)
        print("\nKoriLM :", answer)

# =====================
# Main
# =====================
if __name__ == "__main__":
    chat()

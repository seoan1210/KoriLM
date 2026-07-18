import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from tqdm import tqdm


# =========================
# Config
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 16000

BLOCK_SIZE = 256

BATCH_SIZE = 20

EPOCHS = 1

LR = 3e-4


# 약 50M 파라미터 모델

N_LAYER = 8
N_HEAD = 8
N_EMBD = 512


DATA_FILE = "data.txt"

TOKENIZER = "kori_tokenizer"


# 이어 학습 여부
RESUME = True


# =========================
# Tokenizer
# =========================


def make_tokenizer():

    if os.path.exists(TOKENIZER + ".model"):
        return


    print("Tokenizer 생성")


    spm.SentencePieceTrainer.train(

        input=DATA_FILE,

        model_prefix=TOKENIZER,

        vocab_size=VOCAB_SIZE,

        character_coverage=1.0,

        model_type="unigram"

    )



def load_tokenizer():

    sp = spm.SentencePieceProcessor()

    sp.load(
        TOKENIZER + ".model"
    )

    return sp



# =========================
# Dataset
# =========================


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, sp):

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            text=f.read()


        ids=sp.encode(text)


        self.data=torch.tensor(
            ids,
            dtype=torch.long
        )



    def __len__(self):

        return len(self.data)-BLOCK_SIZE



    def __getitem__(self,index):

        x=self.data[
            index:index+BLOCK_SIZE
        ]

        y=self.data[
            index+1:index+BLOCK_SIZE+1
        ]


        return x,y



# =========================
# GPT Model
# =========================


class CausalSelfAttention(nn.Module):

    def __init__(self):

        super().__init__()


        self.attn=nn.MultiheadAttention(

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



    def forward(self,x):

        size=x.size(1)


        mask=self.mask[
            :size,
            :size
        ]


        out,_=self.attn(

            x,

            x,

            x,

            attn_mask=~mask.bool()

        )


        return out




class Block(nn.Module):

    def __init__(self):

        super().__init__()


        self.ln1=nn.LayerNorm(
            N_EMBD
        )


        self.attn=CausalSelfAttention()


        self.ln2=nn.LayerNorm(
            N_EMBD
        )


        self.mlp=nn.Sequential(

            nn.Linear(
                N_EMBD,
                N_EMBD*4
            ),

            nn.GELU(),

            nn.Linear(
                N_EMBD*4,
                N_EMBD
            )

        )



    def forward(self,x):

        x=x+self.attn(
            self.ln1(x)
        )


        x=x+self.mlp(
            self.ln2(x)
        )


        return x




class KoriLM(nn.Module):

    def __init__(self):

        super().__init__()


        self.token_embedding=nn.Embedding(

            VOCAB_SIZE,

            N_EMBD

        )


        self.position_embedding=nn.Embedding(

            BLOCK_SIZE,

            N_EMBD

        )


        self.blocks=nn.Sequential(

            *[

                Block()

                for _ in range(N_LAYER)

            ]

        )


        self.ln=nn.LayerNorm(
            N_EMBD
        )


        self.head=nn.Linear(

            N_EMBD,

            VOCAB_SIZE

        )



    def forward(self,x):

        B,T=x.shape


        pos=torch.arange(

            T,

            device=x.device

        )


        x=(

            self.token_embedding(x)

            +

            self.position_embedding(pos)

        )


        x=self.blocks(x)


        x=self.ln(x)


        logits=self.head(x)


        return logits




# =========================
# Train
# =========================


def train():


    make_tokenizer()


    sp=load_tokenizer()


    dataset=TextDataset(sp)



    loader=torch.utils.data.DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )



    model=KoriLM().to(DEVICE)



    optimizer=torch.optim.AdamW(

        model.parameters(),

        lr=LR

    )


    step=0



    print(

        "Parameter:",

        sum(
            p.numel()
            for p in model.parameters()
        ) / 1e6,

        "M"

    )



    # =========================
    # Resume
    # =========================


    if RESUME and os.path.exists("checkpoint.pt"):


        print(
            "체크포인트 발견 - 이어서 학습"
        )


        checkpoint=torch.load(

            "checkpoint.pt",

            map_location=DEVICE

        )


        model.load_state_dict(

            checkpoint["model"]

        )


        optimizer.load_state_dict(

            checkpoint["optimizer"]

        )


        step=checkpoint["step"]



        print(

            "현재 step:",

            step

        )



    # =========================
    # Training Loop
    # =========================


    model.train()



    for epoch in range(EPOCHS):


        bar=tqdm(loader)



        for x,y in bar:


            x=x.to(DEVICE)

            y=y.to(DEVICE)



            logits=model(x)



            loss=F.cross_entropy(

                logits.view(

                    -1,

                    VOCAB_SIZE

                ),

                y.view(-1)

            )



            optimizer.zero_grad()


            loss.backward()


            optimizer.step()



            step+=1



            bar.set_description(

                f"step {step} loss {loss.item():.4f}"

            )



            # =====================
            # Save checkpoint
            # =====================


            if step % 1000 == 0:


                torch.save(

                    {

                        "model":
                            model.state_dict(),


                        "optimizer":
                            optimizer.state_dict(),


                        "step":
                            step

                    },


                    "checkpoint.pt"

                )


                print(

                    f"\n💾 체크포인트 저장 완료 : {step} step"

                )




    # =========================
    # Final Save
    # =========================


    torch.save(

        {

            "model":
                model.state_dict(),


            "step":
                step

        },


        "KoriLM.pt"

    )


    print(
        "최종 모델 저장 완료"
    )





# =========================
# Main
# =========================


if __name__=="__main__":


    if len(sys.argv)<2:


        print(
"""
사용법:

학습:
python train.py train

"""
        )


    elif sys.argv[1]=="train":


        train()

import torch
import tiktoken

from gpt2_124m import GPTModel
from create_dataloader import train_dataset

# 事前に定義済みのGPTModelなどの必要なクラス，関数があると仮定します

def build_model():
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = {
        "vocab_size": 50257,          # 1
        "context_length": 1024,       # 2
        "drop_rate": 0.0,             # 3
        "qkv_bias": True              # 4
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    # 学習時と同じモデル構造を生成
    model = GPTModel(BASE_CONFIG)
    
    # 学習時に置き換えた分類用のheadを再定義
    num_classes = 2
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], 
        out_features=num_classes
    )
    
    return model

def load_model(checkpoint_path, device):
    model = build_model()
    # checkpointからパラメータを読み込み
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)          #1
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(              #2
        max_length, supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids))    #3

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)              #4

    with torch.no_grad():                                #5
        logits = model(input_tensor)[:, -1, :]     #6
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"     #7


# PYTHONPATH=~/rnd/projects/build_a_llm uv run inference.py
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "spam_classifier.pth"
    
    # モデルの読み込み
    model = load_model(checkpoint_path, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
       #"Hey, I want to become a king. Then I'll give you a lot of money"
       #"here is a promissing real estate investment opportunity"
       #" that you should take a look at. Let me know if you're interested."
       #"you are a winner. you have been specially selected to receive $1000 cash"
       #"Win a £1000 cash prize or a prize worth $100 "
       #"Win a £1000 cash prize or a prize worth £5000"
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

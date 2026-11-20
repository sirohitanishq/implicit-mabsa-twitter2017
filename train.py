import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import models
from transformers import AutoModel, get_linear_schedule_with_warmup

from dataset import get_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def resnet_backbone():
    r = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    return nn.Sequential(*list(r.children())[:-2])


class TextModel(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(768, n)

    def forward(self, b):
        h = self.bert(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state[:, 0]
        return self.fc(self.drop(h))


class ImageModel(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.backbone = resnet_backbone()
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(2048, n)

    def forward(self, b):
        f = self.backbone(b["image"]).mean((2, 3))
        return self.fc(self.drop(f))


class ConcatModel(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.backbone = resnet_backbone()
        self.fc = nn.Sequential(
            nn.Linear(768 + 2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, n))

    def forward(self, b):
        t = self.bert(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state[:, 0]
        v = self.backbone(b["image"]).mean((2, 3))
        return self.fc(torch.cat([t, v], dim=1))


class CrossAttnModel(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.backbone = resnet_backbone()
        self.proj = nn.Linear(2048, 768)
        self.attn = nn.MultiheadAttention(768, 8, batch_first=True)
        self.ln = nn.LayerNorm(768)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(768, n)

    def forward(self, b):
        h = self.bert(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        v = self.proj(self.backbone(b["image"]).flatten(2).transpose(1, 2))
        a, _ = self.attn(h, v, v)
        h = self.ln(h + a)
        return self.fc(self.drop(h[:, 0]))


MODELS = {
    "text": (TextModel, "text"),
    "image": (ImageModel, "image"),
    "concat": (ConcatModel, "multimodal"),
    "cross": (CrossAttnModel, "multimodal"),
}


def to_device(b):
    return {k: v.to(DEVICE) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = to_device(b)
        ps.extend(model(b).argmax(1).cpu().tolist())
        ys.extend(b["label"].cpu().tolist())
    acc = accuracy_score(ys, ps)
    p, r, f, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    return {"acc": acc, "precision": p, "recall": r, "f1": f}


def run(name, aspect_mode, epochs, batch_size, workers, seed):
    seed_all(seed)
    cls, mode = MODELS[name]
    loaders = get_loaders(mode, aspect_mode, batch_size, workers)
    model = cls().to(DEVICE)

    pre = [p for n, p in model.named_parameters() if n.startswith(("bert", "backbone"))]
    new = [p for n, p in model.named_parameters() if not n.startswith(("bert", "backbone"))]
    groups = [{"params": pre, "lr": 2e-5}]
    if new:
        groups.append({"params": new, "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = epochs * len(loaders["train"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    ckpt = f"best_{name}_{aspect_mode}_s{seed}.pt"
    best_f1 = -1
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for b in loaders["train"]:
            b = to_device(b)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_fn(model(b), b["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        dev = evaluate(model, loaders["dev"])
        print(f"[{name}/{aspect_mode}/s{seed}] epoch {ep}/{epochs} loss {tot/len(loaders['train']):.4f} "
              f"dev acc {dev['acc']:.4f} f1 {dev['f1']:.4f} ({time.time()-t0:.0f}s)")
        if dev["f1"] > best_f1:
            best_f1 = dev["f1"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    test = evaluate(model, loaders["test"])
    print(f"[{name}/{aspect_mode}/s{seed}] TEST acc {test['acc']:.4f} P {test['precision']:.4f} "
          f"R {test['recall']:.4f} F1 {test['f1']:.4f}")

    with open("results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["model", "aspect_mode", "seed", "accuracy", "precision", "recall", "f1"])
        w.writerow([name, aspect_mode, seed, f"{test['acc']:.4f}", f"{test['precision']:.4f}",
                    f"{test['recall']:.4f}", f"{test['f1']:.4f}"])

    del model, opt, loaders
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    ap.add_argument("--aspect", default="mask", choices=["explicit", "mask", "remove"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = ap.parse_args()
    print("device:", DEVICE)
    names = list(MODELS) if a.model == "all" else [a.model]
    for s in a.seeds:
        for n in names:
            run(n, a.aspect, a.epochs, a.batch_size, a.workers, s)


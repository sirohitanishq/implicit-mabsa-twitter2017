import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

ROOT = Path(r"C:\Users\Tanishq\Programming\NLP")
TEXT_DIR = ROOT / "Twitter_17"
IMG_DIR = ROOT / "data" / "IJCAI2019_data" / "IJCAI2019_data" / "twitter2017_images"
EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def find_split(split):
    hits = list(TEXT_DIR.rglob(f"{split}.tsv"))
    hits = [h for h in hits if "twitter2015" not in str(h).lower()]
    if not hits:
        raise FileNotFoundError(f"{split}.tsv not found under {TEXT_DIR}")
    return hits[0]


def build_image_index(img_dir=IMG_DIR):
    index = {}
    for p in Path(img_dir).rglob("*"):
        if p.suffix.lower() in EXTS:
            index.setdefault(p.name, p)
    return index


def read_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for r in reader:
            if len(r) < 5:
                continue
            _, label, image_id, sentence, aspect = r[:5]
            rows.append((int(label), image_id.strip(), sentence.strip(), aspect.strip()))
    return rows


class MABSADataset(Dataset):
    """
    mode: 'text', 'image', or 'multimodal'
    aspect_mode: 'explicit' (aspect shown), 'mask' (aspect -> [MASK]), 'remove' (aspect deleted)
    Labels: 0 = negative, 1 = neutral, 2 = positive
    """

    def __init__(self, split, mode="multimodal", aspect_mode="mask",
                 tokenizer_name="bert-base-uncased", max_len=64, train=False):
        self.rows = read_tsv(find_split(split))
        self.mode = mode
        self.aspect_mode = aspect_mode
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.img_index = build_image_index()
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), norm])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(), norm])

    def __len__(self):
        return len(self.rows)

    def _text(self, sentence, aspect):
        if self.aspect_mode == "explicit":
            return sentence.replace("$T$", aspect)
        if self.aspect_mode == "mask":
            return sentence.replace("$T$", self.tok.mask_token)
        return sentence.replace("$T$", "").replace("  ", " ")

    def _image(self, image_id):
        p = self.img_index.get(image_id)
        if p is None:
            return torch.zeros(3, 224, 224)
        try:
            return self.tf(Image.open(p).convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, i):
        label, image_id, sentence, aspect = self.rows[i]
        item = {"label": torch.tensor(label, dtype=torch.long)}
        if self.mode in ("text", "multimodal"):
            if self.aspect_mode == "explicit":
                enc = self.tok(sentence.replace("$T$", aspect), aspect,
                                padding="max_length", truncation=True,
                                max_length=self.max_len, return_tensors="pt")
            else:
                enc = self.tok(self._text(sentence, aspect), padding="max_length",
                                truncation=True, max_length=self.max_len,
                                return_tensors="pt")
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        if self.mode in ("image", "multimodal"):
            item["image"] = self._image(image_id)
        return item


def get_loaders(mode="multimodal", aspect_mode="mask", batch_size=32, num_workers=0):
    sets = {
        "train": MABSADataset("train", mode, aspect_mode, train=True),
        "dev": MABSADataset("dev", mode, aspect_mode),
        "test": MABSADataset("test", mode, aspect_mode),
    }
    return {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"),
                      num_workers=num_workers)
        for k, v in sets.items()
    }


if __name__ == "__main__":
    print("train.tsv:", find_split("train"))
    idx = build_image_index()
    print("images indexed:", len(idx))
    for s in ("train", "dev", "test"):
        rows = read_tsv(find_split(s))
        missing = sum(1 for r in rows if r[1] not in idx)
        print(f"{s}: {len(rows)} samples, missing images: {missing}")
    print("sample row:", read_tsv(find_split("train"))[0])
    ds = MABSADataset("train", "multimodal", "mask")
    b = ds[0]
    print({k: tuple(v.shape) for k, v in b.items()})
    print(ds.tok.decode(b["input_ids"], skip_special_tokens=False)[:150])
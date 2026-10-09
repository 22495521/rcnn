import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes

# torchvision references/detection 官方輔助檔
# train_one_epoch：訓練一個 epoch，第 0 個 epoch 會 warmup
# evaluate       ：用 pycocotools 算 COCO mAP
import utils
from engine import evaluate, train_one_epoch


# ============================================================
# 參數設定
# ============================================================
DATA_DIR = Path("./vehicles_images")
OUTPUT_DIR = Path("./outputs")

# 0 保留給背景，車種從 1 開始
CLASSES = ["__background__", "Bus", "Car", "Motorcycle", "Pickup", "Truck"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 6 = 背景 + 5 種車

# 畫框用的顏色，每個類別一種
CLASS_COLORS = {"Bus": "orange", "Car": "lime", "Motorcycle": "cyan", "Pickup": "magenta", "Truck": "yellow"}

NUM_EPOCHS = 15
BATCH_SIZE = 5
SCORE_THRESHOLD = 0.5  # 統一門檻（目前改用下面的 CLASS_THRESHOLD）
# 依類別設定門檻，Motorcycle、Pickup、Truck 分數普遍偏低所以調低
CLASS_THRESHOLD = {"Bus": 0.5, "Car": 0.5, "Motorcycle": 0.23, "Pickup": 0.2, "Truck": 0.2}
SEED = 42


# ============================================================
# 資料
# ============================================================
class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_path, transforms=None):
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        # 依檔名分組：{filename: [(x1, y1, x2, y2, label), ...]}
        self.annotations = defaultdict(list)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                box = [float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                self.annotations[row["filename"]].append((*box, CLASS_TO_IDX[row["class"]]))

        # 圖片清單直接用 CSV 裡出現過的檔名，排序確保每次順序一致
        self.imgs = sorted(self.annotations)

    def __getitem__(self, idx):
        fname = self.imgs[idx]
        # 讀圖：shape 為 [3, H, W]，dtype uint8
        img = read_image(str(self.img_dir / fname), mode=ImageReadMode.RGB)

        anns = self.annotations[fname]
        boxes = torch.tensor([a[:4] for a in anns], dtype=torch.float32)
        labels = torch.tensor([a[4] for a in anns], dtype=torch.int64)

        target = {
            # 用 tv_tensors.BoundingBoxes 包裝，做翻轉等資料增強時框會跟著圖一起轉換
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        img = tv_tensors.Image(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transforms():
    """回傳 (train_tf, test_tf)"""
    train_tf = T.Compose([
        T.RandomHorizontalFlip(0.5),          # 50% 機率水平翻轉，框會自動跟著翻
        T.ToDtype(torch.float, scale=True),   # uint8 [0, 255] → float32 [0, 1]
        T.ToPureTensor(),                     # tv_tensors 轉回一般 tensor
    ])

    test_tf = T.Compose([
        T.ToDtype(torch.float, scale=True),
        T.ToPureTensor(),
    ])

    return train_tf, test_tf


def get_dataloaders():
    """回傳 (data_loader, data_loader_test, dataset, dataset_test)"""
    train_tf, test_tf = get_transforms()

    # 資料集已分好 train/、test/，不用再自己切
    dataset = VehicleDataset(DATA_DIR / "train", DATA_DIR / "train_labels.csv", train_tf)
    dataset_test = VehicleDataset(DATA_DIR / "test", DATA_DIR / "test_labels.csv", test_tf)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn
    )

    return data_loader, data_loader_test, dataset, dataset_test


def show_sample(dataset):
    """檢查一筆資料：印出 target 內容並畫出標註框"""
    img, target = dataset[0]
    print("image:", img.shape, img.dtype)
    for k, v in target.items():
        print(f"{k}:", v.shape if isinstance(v, torch.Tensor) else v)

    names = [CLASSES[i] for i in target["labels"].tolist()]
    drawn = draw_bounding_boxes(
        (img * 255).to(torch.uint8), target["boxes"], names, colors=[CLASS_COLORS[n] for n in names], width=2
    )
    plt.figure(figsize=(7, 7))
    plt.imshow(drawn.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


# ============================================================
# 模型
# ============================================================
def get_model(num_classes):
    """載入 COCO 預訓練的 Faster R-CNN，換掉最後的預測頭"""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # 取得原本分類頭的輸入維度（1024）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 換成新的預測頭：輸出 num_classes 個類別分數 + 每類 4 個框座標偏移
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# 訓練與評估
# ============================================================
def train(model, data_loader, data_loader_test, device):
    """訓練 NUM_EPOCHS 個 epoch，每個 epoch 結束跑一次 COCO mAP"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 每 3 個 epoch 學習率乘 0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        # print_freq=10：每 10 個 iteration 印一次 loss
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    print("Finished Training")


@torch.no_grad()
def filter_prediction(pred):
    """依類別門檻過濾，再做跨類別 NMS，回傳 (boxes, scores, labels)"""
    thr = torch.tensor([CLASS_THRESHOLD[CLASSES[l]] for l in pred["labels"].tolist()],
                       device=pred["scores"].device)
    keep = pred["scores"] > thr
    # keep = pred["scores"] > SCORE_THRESHOLD
    boxes, scores, labels = pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]

    # 不分類別去重：重疊超過 0.5 的框只留分數最高的
    idx = nms(boxes, scores, iou_threshold=0.5)
    return boxes[idx], scores[idx], labels[idx]


@torch.no_grad()
def mean_score(model, dataset_test, device):
    """整個測試集的偵測框數與平均分數"""
    model.eval()
    all_scores = []

    for img, _ in dataset_test:
        pred = model([img.to(device)])[0]
        _, scores, _ = filter_prediction(pred)
        all_scores += scores.tolist()

    print(f"共偵測到 {len(all_scores)} 個框，平均分數 {sum(all_scores) / len(all_scores):.4f}")


# ============================================================
# 預測與視覺化
# ============================================================
@torch.no_grad()
def predict_and_show(model, image_path, device, save_path=None):
    """單張圖片預測，畫框顯示並存檔"""
    model.eval()
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    _, test_tf = get_transforms()
    x = test_tf(image).to(device)
    pred = model([x])[0]

    boxes, scores, label_ids = filter_prediction(pred)
    boxes = boxes.cpu()

    names = [CLASSES[i] for i in label_ids.tolist()]
    labels = [f"{n}: {s:.2f}" for n, s in zip(names, scores.tolist())]

    drawn = draw_bounding_boxes(image, boxes, labels, colors=[CLASS_COLORS[n] for n in names], width=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(drawn.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

    if save_path is not None:
        F.to_pil_image(drawn).save(save_path)


# ============================================================
# 模型存取
# ============================================================
def save_model(model, model_path):
    """儲存模型權重"""
    torch.save(model.state_dict(), model_path)
    print(f"模型已儲存: {model_path}")


def load_model(model, model_path, device):
    """載入模型權重"""
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"已載入模型: {model_path}")


# ============================================================
# 主程式
# ============================================================
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    data_loader, data_loader_test, dataset, dataset_test = get_dataloaders()
    show_sample(dataset)

    model = get_model(NUM_CLASSES).to(device)

    # 已經有存好的權重就直接載入，沒有才訓練並存檔
    model_path = OUTPUT_DIR / "fasterrcnn_vehicles.pth"
    if os.path.exists(model_path):
        load_model(model, model_path, device)
    else:
        train(model, data_loader, data_loader_test, device)
        save_model(model, model_path)

    # 前 3 張測試圖預測並存檔
    for i in range(3):
        test_img_path = dataset_test.img_dir / dataset_test.imgs[i]
        predict_and_show(model, test_img_path, device, OUTPUT_DIR / f"prediction_{i}.png")

    mean_score(model, dataset_test, device)


if __name__ == "__main__":
    main()

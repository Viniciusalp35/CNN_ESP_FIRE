import os
import cv2
import glob
import numpy as np
from pathlib import Path

SOURCE_ROOT = "./fire_data"

DEST_ROOT = "./fire_data_processed"

MIN_SIZE = 32


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    return interArea / float(boxAArea)


def process_dataset():
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = os.path.join(SOURCE_ROOT, split, "images")
        lbl_dir = os.path.join(SOURCE_ROOT, split, "labels")

        dest_fire = os.path.join(DEST_ROOT, split, "1")
        dest_bg = os.path.join(DEST_ROOT, split, "0")
        os.makedirs(dest_fire, exist_ok=True)
        os.makedirs(dest_bg, exist_ok=True)

        # Pega imagens
        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(
            os.path.join(img_dir, "*.png")
        )

        print(f"Processando {split}: {len(img_files)} imagens...")

        for img_path in img_files:
            filename = Path(img_path).stem
            lbl_path = os.path.join(lbl_dir, filename + ".txt")

            if not os.path.exists(lbl_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            # Lê as bounding boxes do fogo
            fire_boxes = []  # Lista de [x1, y1, x2, y2]

            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    try:
                        _, x_c, y_c, w, h = map(float, parts)
                    except:
                        continue  # Pula linhas inválidas
                    # Converte para pixels
                    w_px, h_px = int(w * w_img), int(h * h_img)
                    x_c_px, y_c_px = int(x_c * w_img), int(y_c * h_img)

                    x1 = max(0, x_c_px - w_px // 2)
                    y1 = max(0, y_c_px - h_px // 2)
                    x2 = min(w_img, x1 + w_px)
                    y2 = min(h_img, y1 + h_px)

                    fire_boxes.append([x1, y1, x2, y2])

                    # Recorta o fogo e salva
                    roi_fire = img[y1:y2, x1:x2]
                    if roi_fire.size > 0 and w_px > MIN_SIZE and h_px > MIN_SIZE:
                        save_name = f"{filename}_fire_{i}.jpg"
                        cv2.imwrite(os.path.join(dest_fire, save_name), roi_fire)

            for j in range(3):
                bg_w = np.random.randint(64, 256)
                bg_h = np.random.randint(64, 256)

                if bg_w >= w_img or bg_h >= h_img:
                    continue

                # Posição aleatória
                bg_x1 = np.random.randint(0, w_img - bg_w)
                bg_y1 = np.random.randint(0, h_img - bg_h)
                bg_x2 = bg_x1 + bg_w
                bg_y2 = bg_y1 + bg_h

                # Verifica se colide com algum fogo
                bg_box = [bg_x1, bg_y1, bg_x2, bg_y2]
                colisao = False
                for fbox in fire_boxes:
                    # Se a intersecção for maior que 0, considera colisão
                    if iou(bg_box, fbox) > 0:
                        colisao = True
                        break

                if not colisao:
                    roi_bg = img[bg_y1:bg_y2, bg_x1:bg_x2]
                    save_name = f"{filename}_bg_{j}.jpg"
                    cv2.imwrite(os.path.join(dest_bg, save_name), roi_bg)


if __name__ == "__main__":
    process_dataset()

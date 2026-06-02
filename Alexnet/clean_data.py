import os
from PIL import Image

data_dir = "data"
removed = 0

for split in ["train", "val"]:
    for category in ["cat", "dog"]:
        folder = os.path.join(data_dir, split, category)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath)
                img.verify()  # 验证图像完整性
                # verify 后需要重新打开
                img = Image.open(fpath)
                img.load()
            except Exception:
                os.remove(fpath)
                removed += 1
                print(f"已删除损坏文件: {fpath}")

print(f"\n清理完成，共删除 {removed} 个损坏文件")

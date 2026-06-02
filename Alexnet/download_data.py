import os
import shutil


def main():
    # 当前目录下的 PetImages 文件夹
    petimages_dir = "PetImages"
    if not os.path.isdir(petimages_dir):
        print(f"错误: 找不到 {petimages_dir} 目录，请将其放在脚本同级目录下")
        return

    data_dir = "data"
    train_cat_dir = os.path.join(data_dir, "train", "cat")
    train_dog_dir = os.path.join(data_dir, "train", "dog")
    val_cat_dir = os.path.join(data_dir, "val", "cat")
    val_dog_dir = os.path.join(data_dir, "val", "dog")

    # 清空旧数据
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    for d in [train_cat_dir, train_dog_dir, val_cat_dir, val_dog_dir]:
        os.makedirs(d, exist_ok=True)

    split_ratio = 0.8

    for category in ["Cat", "Dog"]:
        src_dir = os.path.join(petimages_dir, category)
        target_label = "cat" if category == "Cat" else "dog"

        images = sorted(os.listdir(src_dir))
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_target = train_cat_dir if target_label == "cat" else train_dog_dir
        val_target = val_cat_dir if target_label == "cat" else val_dog_dir

        for img in train_images:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(train_target, img))

        for img in val_images:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(val_target, img))

        print(f"{category}: {len(train_images)} 训练 / {len(val_images)} 验证")

    train_total = sum(len(os.listdir(d)) for d in [train_cat_dir, train_dog_dir])
    val_total = sum(len(os.listdir(d)) for d in [val_cat_dir, val_dog_dir])
    print(f"\n数据集准备完成! 训练集: {train_total} 张, 验证集: {val_total} 张")


if __name__ == "__main__":
    main()

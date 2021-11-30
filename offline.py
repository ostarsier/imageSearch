from pathlib import Path

import numpy as np
from PIL import Image

from feature_extractor import FeatureExtractor

# 批量抽取img目录下所有图片向量
if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.name + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify

from feature_extractor import FeatureExtractor

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Read image features
fe = FeatureExtractor()
features = []
img_paths = []


def init_feature():
    global features
    for feature_path in Path("./static/feature").glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/img") / (feature_path.stem))
    features = np.array(features)


def add_feature(feature_path, feature):
    global features
    features = np.row_stack((features, feature))

    img_paths.append(Path("./static/img") / (feature_path.stem))


init_feature()


# 保存上传图片向量
@app.route('/img/save', methods=['POST'])
def img_save():
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save image
        img = Image.open(file.stream)
        filename = uuid.uuid4().hex + "_" + file.filename
        uploaded_img_path = "./static/img/" + filename
        img.save(uploaded_img_path)

        feature = fe.extract(img=Image.open(uploaded_img_path))
        feature_path = Path("./static/feature") / (filename + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
        add_feature(feature_path, feature)
        return 'success'
    else:
        return '只能上传jpg、png、jpeg格式的图片'


@app.route('/img/search', methods=['POST'])
def img_search():
    file = request.files['file']
    # Save query image
    img = Image.open(file.stream)  # PIL image
    uploaded_img_path = "static/uploaded/" + uuid.uuid4().hex + "_" + file.filename
    img.save(uploaded_img_path)

    # Run search
    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:30]  # Top 30 results
    scores = [str(img_paths[id]) for id in ids]
    return jsonify(scores)


# 首页
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + uuid.uuid4().hex + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)

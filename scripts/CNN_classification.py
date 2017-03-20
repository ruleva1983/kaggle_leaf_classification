import pandas as pd
import os
import sys

sys.path.append('../source/')

from learn import CNN_Classifier
from preprocessing import augment_data_set


def run():
    path = "../data/"

    data_train = pd.read_csv(os.path.join(path, "train.csv"))
    data_train = data_train.set_index(["id"], drop=True)

    data_test = pd.read_csv(os.path.join(path, "test.csv"))
    data_test = data_test.set_index(["id"], drop=True)

    dir_path = "../data/images/"
    files = os.listdir(dir_path)
    X_images, X_features, y = augment_data_set(files, data_train, 20, 0.9, 0.9, shuffle=True, IMAGE_DIM=200)
    structure = [{"type": "conv", "params": {"patch_x": 10, "patch_y": 10, "depth": 64, "channels": 1}},
                 {"type": "pool", "params": {"side": 2, "stride": 2, "pad": "SAME"}},
                 {"type": "conv", "params": {"patch_x": 6, "patch_y": 6, "depth": 32, "channels": 64}},
                 {"type": "pool", "params": {"side": 2, "stride": 2, "pad": "SAME"}},
                 {"type": "dense", "params": {"n_input": 2500 * 32, "n_neurons": 500}}]

    classifier = CNN_Classifier(structure=structure, nb_classes=99, img_rows=200, img_cols=200, nb_hidden=(1024,),
                                nb_features=192)
    classifier.fit({"image": X_images, "features": X_features}, y, batch_size=64, nb_epochs=1000, logging_info=20)


if __name__ == "__main__":
    run()
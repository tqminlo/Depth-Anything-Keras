import numpy as np
from keras.layers import *
import keras
from keras.models import Model
import cv2
from dat_model import DAT


model = keras.models.load_model("saved/dat.h5")


def load_image(filepath):
    image = cv2.imread(filepath)  # H, W, C
    orig_shape = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    # image = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])({"image": image})["image"]
    # image = transform({"image": image})["image"]
    image = cv2.resize(image, (518, 518))
    image = np.expand_dims(image, axis=0)  # B, C, H, W

    return image, orig_shape


# def check_infer_layer(img_path, check_layer_name):
#     image, orig_shape = load_image(img_path)
#     inp1 = np.load("saved/inp1.npy")
#     inp2 = np.load("saved/inp2.npy")
#     # print(inp1.shape)
#     # print(inp2.shape)
#     inp = [image, inp1, inp2]
#
#     check_layer = model.get_layer(check_layer_name)
#     check_model = Model(model.input, check_layer.output)
#     check_out = check_model.predict(inp)[0]
#     print(check_out)
#     print(check_out.shape)


def infer(img_path):
    image, (orig_h, orig_w) = load_image(img_path)
    inp1 = np.load("saved/inp1.npy")
    inp2 = np.load("saved/inp2.npy")
    inp = [image, inp1, inp2]

    depth = model.predict(inp)
    depth = depth[0]
    # print(depth)
    # print(depth.shape)

    depth = cv2.resize(depth, (orig_w, orig_h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    margin_width = 50
    caption_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    split_region = np.ones((orig_h, margin_width, 3), dtype=np.uint8) * 255
    combined_results = cv2.hconcat([cv2.imread(img_path), split_region, depth_color])

    caption_space = (
            np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8)
            * 255
    )
    captions = ["Raw image", "Depth Anything"]
    segment_width = orig_w + margin_width
    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (orig_w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(
            caption_space,
            caption,
            (text_x, 40),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    final_result = cv2.vconcat([caption_space, combined_results])

    cv2.imshow("depth", final_result)
    cv2.waitKey(0)


if __name__ == "__main__":
    infer("test/demo1.png")

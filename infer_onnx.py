import onnx, onnxruntime
import cv2
import numpy as np
from onnx import helper


def load_image(filepath):
    image = cv2.imread(filepath)  # H, W, C
    orig_shape = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    # image = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])({"image": image})["image"]
    # image = transform({"image": image})["image"]
    image = cv2.resize(image, (518, 518))
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    image = image[None]  # B, C, H, W

    return image, orig_shape


def infer(img_path):
    onnx_name = "saved/depth_anything_vits14_1x3x518x518.onnx"
    image, (orig_h, orig_w) = load_image(img_path)
    print(type(image))
    print(np.min(image), np.max(image))
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_name, providers=EP_list)
    # print(session.get_inputs()[0].name)    ### "input"
    ort_inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(["output"], ort_inputs)    # infer model
    depth = outputs[0]
    # check_output_1 = outputs[1]
    # print(check_output_1[0])

    # Visualization
    viz = True
    if viz:
        depth = cv2.resize(depth[0, 0], (orig_w, orig_h))
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


def check_infer_layer(img_path, check_output_name):
    intermediate_layer_value_info = helper.ValueInfoProto()
    onnx_name = "saved/depth_anything_vits14_1x3x518x518.onnx"
    onnx_new_path = "saved/depth_anything_vits14_extra_all_out.onnx"
    model = onnx.load(onnx_name)

    append_outputs = [check_output_name] if type(check_output_name) is str else check_output_name
    for output_name in append_outputs:
        intermediate_layer_value_info.name = output_name
        model.graph.output.append(intermediate_layer_value_info)

    onnx.save(model, onnx_new_path)
    print("---")

    image, (orig_h, orig_w) = load_image(img_path)
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_new_path, providers=EP_list)
    ort_inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(append_outputs, ort_inputs)  # infer model
    for output in outputs:
        print(output[0][0][0])
        print("-----")


if __name__ == "__main__":
    # check_infer_layer("test/demo1.png",
    #                   ["/depth_head/refinenet3/resConfUnit2/Add_output_0", "/depth_head/refinenet3/Resize_output_0"])
    infer("test/demo1.png")
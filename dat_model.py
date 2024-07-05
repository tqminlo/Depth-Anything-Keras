import numpy as np
from keras.layers import *
import keras
from keras.models import Model
import onnx
from onnx import numpy_helper
import tensorflow as tf


class DAT:
    def __init__(self):
        self.size = 518

    def attn_block(self, i, x):
        x1 = LayerNormalization(epsilon=9.999999974752427e-7, name=f"._blocks.{i}._norm1._LayerNormalization")(x)
        x1 = Dense(units=1152, name=f"._blocks.{i}._attn._qkv._MatMulAdd")(x1)
        x1 = Reshape(target_shape=(x1.shape[1], 3, 6, 64))(x1)
        x1 = Permute(dims=(2, 3, 1, 4))(x1)
        q = x1[:, 0, :, :, :]       # (None, 6, 1370, 64)
        k = x1[:, 1, :, :, :]       # (None, 6, 1370, 64)
        v = x1[:, 2, :, :, :]       # (None, 6, 1370, 64)
        k = Permute(dims=(1, 3, 2))(k)      # (None, 6, 64, 1370)
        attn = q @ k                # (None, 6, 1370, 1370)
        attn = attn * 0.125
        attn = Softmax()(attn)
        attn = attn @ v
        x1 = Permute(dims=(2, 1, 3))(attn)
        x1 = Reshape(target_shape=(x1.shape[1], 384))(x1)
        x1 = Dense(units=384, name=f"._blocks.{i}._attn._proj._MatMulAddMul")(x1)
        x = Add()([x, x1])

        return x

    def mlp_block(self, i, x):
        x1 = LayerNormalization(epsilon=9.999999974752427e-7, name=f"._blocks.{i}._norm2._LayerNormalization")(x)
        x1 = Dense(units=1536, name=f"._blocks.{i}._mlp._fc1._MatMulAdd")(x1)
        x1 = keras.activations.gelu(x1)
        x1 = Dense(units=384, name=f"._blocks.{i}._mlp._fc2._MatMulAddMul")(x1)
        x = Add()([x, x1])

        return x

    def bottleneck(self, i, scale, c, x):
        x = LayerNormalization(epsilon=9.999999974752427e-7, name=f"._norm_{i}._LayerNormalization")(x)
        x = x[:, 1:, :]
        x = Reshape(target_shape=(37, 37, x.shape[2]))(x)
        x = Conv2D(filters=c, kernel_size=1, strides=1, padding="valid", name=f"._depth_head._projects.{i}._Conv")(x)
        if scale == 4 or scale == 2:
            x = Conv2DTranspose(filters=c, kernel_size=scale, strides=scale, padding="valid",
                                name=f"._depth_head._resize_layers.{i}._ConvTranspose")(x)
        elif scale == 1/2:
            x = Conv2D(filters=c, kernel_size=3, strides=2, padding="same",
                       name=f"._depth_head._resize_layers.{i}._Conv")(x)
        x = Conv2D(64, kernel_size=3, strides=1, padding="same", use_bias=False,
                   name=f"._depth_head._layer{i+1}_rn._Conv")(x)

        return x

    def after_bottleneck(self, i1, i2, x):
        x1 = ReLU()(x)
        x1 = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu",
                    name=f"._depth_head._refinenet{i1}._resConfUnit{i2}._conv1._Conv")(x1)
        x1 = Conv2D(64, kernel_size=3, strides=1, padding="same",
                    name=f"._depth_head._refinenet{i1}._resConfUnit{i2}._conv2._Conv")(x1)
        x = Add()([x1, x])

        return x

    def __call__(self):
        inp = Input(shape=(518, 518, 3))
        pos_embed_concat = Input(shape=(1, 384))
        pos_embed_add = Input(shape=(1370, 384))
        x = Conv2D(384, kernel_size=14, strides=14, padding="valid", name="._patch_embed._proj._Conv")(inp)

        x = Reshape(target_shape=(x.shape[1]*x.shape[2], x.shape[3]))(x)
        x = Concatenate(axis=1)([pos_embed_concat, x])
        x = Add()([x, pos_embed_add])

        for i in range(9):
            x = self.mlp_block(i, self.attn_block(i, x))
        x1 = self.mlp_block(9, self.attn_block(9, x))
        x2 = self.mlp_block(10, self.attn_block(10, x1))
        x3 = self.mlp_block(11, self.attn_block(11, x2))

        x = self.after_bottleneck(i1=1, i2=1, x=self.bottleneck(i=0, scale=4, c=48, x=x))
        x1 = self.after_bottleneck(i1=2, i2=1, x=self.bottleneck(i=1, scale=2, c=96, x=x1))
        x2 = self.after_bottleneck(i1=3, i2=1, x=self.bottleneck(i=2, scale=1, c=192, x=x2))
        x3 = self.after_bottleneck(i1=4, i2=2, x=self.bottleneck(i=3, scale=1/2, c=384, x=x3))

        x3 = Resizing(height=37, width=37, interpolation="bilinear")(x3)
        x3 = Conv2D(filters=64, kernel_size=1, strides=1, padding="valid",
                    name="._depth_head._refinenet4._out_conv._Conv")(x3)
        x2 = Add()([x3, x2])
        x2 = self.after_bottleneck(i1=3, i2=2, x=x2)
        x2 = Resizing(height=74, width=74, interpolation="bilinear")(x2)
        x2 = Conv2D(filters=64, kernel_size=1, strides=1, padding="valid",
                    name="._depth_head._refinenet3._out_conv._Conv")(x2)
        x1 = Add()([x2, x1])
        x1 = self.after_bottleneck(i1=2, i2=2, x=x1)
        x1 = Resizing(height=148, width=148, interpolation="bilinear")(x1)
        x1 = Conv2D(filters=64, kernel_size=1, strides=1, padding="valid",
                    name="._depth_head._refinenet2._out_conv._Conv")(x1)
        x = Add()([x1, x])
        x = self.after_bottleneck(i1=1, i2=2, x=x)
        x = Resizing(height=296, width=296, interpolation="bilinear")(x)
        x = Conv2D(filters=64, kernel_size=1, strides=1, padding="valid",
                   name="._depth_head._refinenet1._out_conv._Conv")(x)

        x = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", name="._depth_head._output_conv1._Conv")(x)
        x = Resizing(height=518, width=518, interpolation="bilinear")(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu",
                   name="._depth_head._output_conv2._output_conv2.0._Conv")(x)
        x = Conv2D(filters=1, kernel_size=1, strides=1, padding="valid", activation="relu",
                   name="._depth_head._output_conv2._output_conv2.2._Conv")(x)

        model = Model([inp, pos_embed_concat, pos_embed_add], x)

        return model


if __name__ == "__main__":
    def map_w_onnx():
        onnx_name = "saved/depth_anything_vits14_1x3x518x518.onnx"
        onnx_model = onnx.load(onnx_name)
        initializers = list(onnx_model.graph.initializer)
        nodes = onnx_model.graph.node
        mapping = {}
        for node in nodes:
            mapping[node.name] = {}

        for initializer in initializers:
            w_name = initializer.name
            print(w_name)
            w = numpy_helper.to_array(initializer)

            for node in nodes:
                if w_name in node.input:
                    print(node.name)
                    mapping[node.name][w_name] = w
                    print("---")
                    # break

        return mapping

    def onnx2keras_dat(mapping, save_path):
        model = DAT()()
        # model.summary()
        for layer in model.layers:
            if "conv" in layer.name.lower():
                name = layer.name
                print(name)
                onnx_name = name.replace("._", "/")
                keys_w = list(mapping[onnx_name].keys())
                print(keys_w)
                kernel = mapping[onnx_name][keys_w[0]]
                kernel = kernel.transpose(2, 3, 1, 0)
                print(kernel.shape)

                if len(keys_w) == 1:
                    convert_w = [kernel]
                else:
                    bias = mapping[onnx_name][keys_w[1]]
                    print(bias.shape)
                    convert_w = [kernel, bias]

                layer.set_weights(convert_w)
                print("---")

            elif "LayerNormalization" in layer.name:
                name = layer.name
                print(name)
                onnx_name = name.replace("._", "/")
                if onnx_name == "/norm_0/LayerNormalization":
                    onnx_name = "/norm/LayerNormalization"
                keys_w = list(mapping[onnx_name].keys())
                print(keys_w)
                gamma = mapping[onnx_name][keys_w[0]]
                beta = mapping[onnx_name][keys_w[1]]
                layer.set_weights([gamma, beta])
                print("---")

            elif "MatMulAdd" in layer.name and "MatMulAddMul" not in layer.name:
                name = layer.name
                print(name)
                onnx_name_matmul = name.replace("._", "/").replace("MatMulAdd", "MatMul")
                onnx_name_add = name.replace("._", "/").replace("MatMulAdd", "Add")
                keys_w_matmul = list(mapping[onnx_name_matmul].keys())
                print(keys_w_matmul)
                keys_w_add = list(mapping[onnx_name_add].keys())
                print(keys_w_add)
                kernel = mapping[onnx_name_matmul][keys_w_matmul[0]]
                bias = mapping[onnx_name_add][keys_w_add[0]]
                layer.set_weights([kernel, bias])

            elif "MatMulAddMul" in layer.name:
                name = layer.name
                print(name)
                onnx_name_matmul = name.replace("._", "/").replace("MatMulAddMul", "MatMul")
                onnx_name_add = name.replace("._", "/").replace("MatMulAddMul", "Add")
                onnx_name_mul = name.replace("._", "/").replace("MatMulAddMul", "Mul")
                onnx_name_mul = onnx_name_mul.replace("attn/proj", "ls1").replace("mlp/fc2", "ls2")
                keys_w_matmul = list(mapping[onnx_name_matmul].keys())
                print(keys_w_matmul)
                keys_w_add = list(mapping[onnx_name_add].keys())
                print(keys_w_add)
                keys_w_mul = list(mapping[onnx_name_mul].keys())
                print(keys_w_mul)
                kernel = mapping[onnx_name_matmul][keys_w_matmul[0]]
                bias = mapping[onnx_name_add][keys_w_add[0]]
                mul = mapping[onnx_name_mul][keys_w_mul[0]]
                kernel = kernel * mul
                bias = bias * mul
                layer.set_weights([kernel, bias])

        model.save(save_path)

    onnx2keras_dat(map_w_onnx(), "saved/dat.h5")



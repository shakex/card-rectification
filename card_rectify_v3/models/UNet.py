import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNet', 'VGGUNet', 'UNetFCN', 'UNetSegNet']


"""
Implementation code for UNet.
"""

class UNet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, input_channel=3, is_batchnorm=True
    ):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # outputs2 = F.interpolate(inputs2, size=[inputs1.size(2), inputs1.size(3)], mode='bilinear', align_corners=True)
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear', align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetUp2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp2, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(out_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs1 = inputs1
        outputs2 = F.interpolate(outputs2, size=[outputs1.size(2), outputs1.size(3)], mode='bilinear', align_corners=True)

        return self.conv(torch.cat([outputs1, outputs2], 1))



"""
Implementation code for UNet(decoder) with VGG16(encoder) (VGGUNet).
"""

class VGGUNet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, input_channel=3, is_batchnorm=True
    ):
        super(VGGUNet, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # downsampling
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv_block4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv_block5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # upsampling
        self.up_concat4 = unetUp2(1024, 512, self.is_deconv)
        self.up_concat3 = unetUp(512, 256, self.is_deconv)
        self.up_concat2 = unetUp(256, 128, self.is_deconv)
        self.up_concat1 = unetUp(128, 64, self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv_block1(inputs)

        conv2 = self.conv_block2(conv1)

        conv3 = self.conv_block3(conv2)

        conv4 = self.conv_block4(conv3)

        center = self.conv_block5(conv4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final



"""
Implementation code for FCN(decoder) with UNet backbone(encoder) (UNetFCN).
"""


class UNetFCN(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, input_channel=3, is_batchnorm=True
    ):
        super(UNetFCN, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.score = unetConv2(filters[4], n_classes, self.is_batchnorm)

        self.score_pool4 = nn.Conv2d(filters[3], n_classes, 1)
        self.score_pool3 = nn.Conv2d(filters[2], n_classes, 1)
        self.score_pool2 = nn.Conv2d(filters[1], n_classes, 1)
        self.score_pool1 = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.maxpool1(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.maxpool2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.maxpool3(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.maxpool4(conv4)

        conv5 = self.center(conv4)
        conv5 = self.maxpool4(conv5)
        score = self.score(conv5)


        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        score_pool2 = self.score_pool2(conv2)
        score_pool1 = self.score_pool1(conv1)
        score = F.interpolate(score, score_pool4.size()[2:])
        # score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.interpolate(score, score_pool3.size()[2:])
        # score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3
        score = F.interpolate(score, score_pool2.size()[2:])
        score += score_pool2
        score = F.interpolate(score, score_pool1.size()[2:])
        score += score_pool1
        out = F.interpolate(score, inputs.size()[2:])
        # out = F.upsample(score, x.size()[2:])

        return out





"""
Implementation code for SegNet(decoder) with UNet backbone(encoder) (UNetSegNet).
"""


class UNetSegNet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, input_channel=3, is_batchnorm=True
    ):
        super(UNetSegNet, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], self.is_batchnorm)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.up5 = segnetUp2(filters[4], filters[3])
        self.up4 = segnetUp2(filters[3], filters[2])
        self.up3 = segnetUp2(filters[2], filters[1])
        self.up2 = segnetUp2(filters[1], filters[0])
        self.up1 = segnetUp2(filters[0], n_classes)

        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        unpool_shape1 = conv1.size()
        conv1, indices_1 = self.maxpool_with_argmax(conv1)

        conv2 = self.conv2(conv1)
        unpool_shape2 = conv2.size()
        conv2, indices_2 = self.maxpool_with_argmax(conv2)

        conv3 = self.conv3(conv2)
        unpool_shape3 = conv3.size()
        conv3, indices_3 = self.maxpool_with_argmax(conv3)

        conv4 = self.conv4(conv3)
        unpool_shape4 = conv4.size()
        conv4, indices_4 = self.maxpool_with_argmax(conv4)

        conv5 = self.center(conv4)
        unpool_shape5 = conv5.size()
        conv5, indices_5 = self.maxpool_with_argmax(conv5)

        up5 = self.up5(conv5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


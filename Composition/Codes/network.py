import torch
import torch.nn as nn
import torch.nn.functional as F

#这是我第一次修改网络主体的尝试！看来pytorch还是很好用的！ 总体来说对于网络的修改和尝试还是算顺利的
#conv2d这个函数我搞的尺寸居然是一次性就对上了！那么看来我模式识别谢老师教的还是很不错的

def build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor):

    out  = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)#这个out的结果就是网络输出的res
    out_list = torch.split(out , 1, dim=1)

    # 分割后的两个张量
    out1 = out_list[0]
    out2 = out_list[1]
    # 这个函数是为了将网络的输出变成最终要的呈现出来的结果的样子
    #这个out就是一个掩膜的掩膜吧 就是一个掩膜的权值一样的东西！

    learned_mask1_1 = (mask1_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*out1
    learned_mask2_1 = (mask2_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*(1-out1)
    stitched_image_1 = (warp1_tensor+1.) * learned_mask1_1 + (warp2_tensor+1.)*learned_mask2_1 - 1.

    learned_mask1_2 = (mask1_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*out2
    learned_mask2_2 = (mask2_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*(1-out2)
    stitched_image_2 = (warp1_tensor+1.) * learned_mask1_2 + (warp2_tensor+1.)*learned_mask2_2 - 1.

    out_dict = {}
    # out_dict.update(learned_mask1=learned_mask1_1, learned_mask2=learned_mask2_1, stitched_image = stitched_image_1)
    out_dict.update(learned_mask1_1=learned_mask1_1, learned_mask2_1=learned_mask2_1, stitched_image_1 = stitched_image_1,
                    learned_mask1_2=learned_mask1_2, learned_mask2_2=learned_mask2_2, stitched_image_2 = stitched_image_2)

    return out_dict


class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(UpBlock, self).__init__()
        #self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.halfChanelConv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):

        x1 = F.interpolate(x1, size = (x2.size()[2], x2.size()[3]), mode='nearest')
        x1 = self.halfChanelConv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class ExchangeConv(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(ExchangeConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1,x2):
        x1 = self.conv(x1)
        x = x1 + x2
        return x
# predict the composition mask of img1
class Network(nn.Module):
    def __init__(self, nclasses=1):
        super(Network, self).__init__()


        self.down1 = DownBlock(3, 32, 1, pool=False)
        self.down2 = DownBlock(32, 64, 2)
        self.down3 = DownBlock(64, 128,3)
        self.down4 = DownBlock(128, 256, 4)
        self.down5 = DownBlock(256, 512, 5)
        self.up1 = UpBlock(512, 256, 4)
        self.exconv1 = ExchangeConv(256,256,1)
        self.up2 = UpBlock(256, 128, 3)
        self.exconv2 = ExchangeConv(128, 128, 1)
        self.up3 = UpBlock(128, 64, 2)
        self.exconv3 = ExchangeConv(64, 64, 1)
        self.up4 = UpBlock(64, 32, 1)


        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, y, m1, m2):


        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        y1 = self.down1(y)
        y2 = self.down2(y1)
        y3 = self.down3(y2)
        y4 = self.down4(y3)
        y5 = self.down5(y4)

        res1_1 = self.up1(x5-y5, x4-y4)
        res2_1 = self.up1(x5 - y5, x4 - y4)
        res1_1_new =self.exconv1(res2_1,res1_1)
        res2_1_new = self.exconv1(res1_1, res2_1)

        res1_2 = self.up2(res1_1_new, x3-y3)
        res2_2 = self.up2(res2_1_new, x3 - y3)

        res1_2_new = self.exconv2(res2_2, res1_2)
        res2_2_new = self.exconv2(res1_2, res2_2)

        res1_3 = self.up3(res1_2_new, x2-y2)
        res2_3 = self.up3(res2_2_new, x2 - y2)

        res1_3_new = self.exconv3(res2_3, res1_3)
        res2_3_new = self.exconv3(res1_3, res2_3)

        res1_4 = self.up4(res1_3_new, x1-y1)
        res2_4 = self.up4(res2_3_new, x1 - y1)
        res1 = self.out(res1_4)
        res2 = self.out(res2_4)
        #res = 0.5 * (res1 + res2)
        res = torch.cat([res1,res2], dim=1)



        # res1 = self.up1(x5-y5, x4-y4)
        # res1 = self.up2(res1, x3-y3)
        # res1 = self.up3(res1, x2-y2)
        # res1 = self.up4(res1, x1-y1)
        # res1 = self.out(res1)#out是一个sigmoid了其实这个就是在计算那种分类的权值的意思了！
        # res2 = self.up1(x5-y5, x4-y4)
        # res2 = self.up2(res2, x3-y3)
        # res2 = self.up3(res2, x2-y2)
        # res2 = self.up4(res2, x1-y1)
        # res2 = self.out(res2)#out是一个sigmoid了其实这个就是在计算那种分类的权值的意思了！



        return res

# model = Network()
# input = torch.randn(1, 3, 128, 128)
# leaned1,learned2=model(input)
# flops, params = profile(model, (input,))
# print('flops: ', flops, 'params: ', params)

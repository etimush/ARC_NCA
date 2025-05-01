import torch

def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], mode='circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
lap = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8, 2.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")


def perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    if mask_n != 0:
        n = x.shape[1]
        padd = torch.zeros((x.shape[0], 3 * mask_n, x.shape[2], x.shape[3]), device="cuda:0")
        obs = perchannel_conv(x[:, 0:n - mask_n], filters)
        return torch.cat((x, obs, padd), dim=1)
    else:
        obs = perchannel_conv(x, filters)
        return torch.cat((x,obs), dim = 1 )



def reduced_perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    x_redu = x[:,0:x.shape[1]-mask_n]
    obs = perchannel_conv(x_redu,filters)
    return torch.cat((x,obs), dim = 1 )




class GeneCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size = 3):
        super().__init__()
        self.chn = chn

        self.w1 = torch.nn.Conv2d(chn + 3*(chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn-gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:,-self.gene_size:,...]

        y = reduced_perception(x, 0)

        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        #xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(x[:, None, 3, ...], 3, 1, 1, ).cuda() > 0.1
        #alive_mask = x[:, None, 3, ...]

        x = x[:,:x.shape[1]-self.gene_size,...] + y * update_mask #* pre_life_mask
        x = torch.cat((x, gene), dim=1)
        return x




class GenePropCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4*chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n,  gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...]

        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        #xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(x[:, None, 3, ...], 3, 1, 1, ).cuda() > 0.1

        gene = gene + y  * update_mask #* pre_life_mask

        x = x[:, :x.shape[1] - self.gene_size, ...]


        x = torch.cat((x, gene), dim=1)
        return x








class MixedCA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=128,  gene_size=3):
    super().__init__()
    self.chn = chn
    self.extras = []
    self.dropout = torch.nn.Dropout2d(p=0.4)
    self.dropout2 = torch.nn.Dropout2d(p=0.4)
    self.gene_size = gene_size
    self.perc = torch.nn.Conv2d(chn, 3 * chn, 3, padding=1, padding_mode="zeros", bias=False)
    self.w1 = torch.nn.Conv2d(4*chn, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn- gene_size, 1, bias=False)
    self.w2.weight.data.zero_()
    self.attent = LocalChannelAttention(4 * chn, 8)
    self.attent2 = LocalChannelAttention(hidden_n, 8)


  def forward(self, x, update_rate=0.5):
    gene = x[:, -self.gene_size:, ...]

    y = self.perc(x)
    y = torch.cat((y,x), dim =1 )
    y = self.attent(y)
    y = self.dropout(y)
    y = self.w1(y)
    y = torch.nn.functional.relu(y)
    y= self.attent2(y)
    y = self.dropout2(y)

    y = self.w2(y)
    b, c, h, w = y.shape
    update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
    pre_life_mask = torch.nn.functional.max_pool2d(x[:,None,3,...], 3, 1, 1).cuda() > 0.1


      # Perform update
    x = x[:, :x.shape[1] - self.gene_size, ...] + y * update_mask #* pre_life_mask


    x = torch.cat((x, gene), dim=1)


    return x



class MixedGenePropCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn
        self.extras = []
        self.attent = LocalChannelAttention(4*chn, 8)
        self.attent2 = LocalChannelAttention(hidden_n, 8)

        self.perc = torch.nn.Conv2d(chn, 3*chn, 3, padding=1, padding_mode="circular", bias=False)
        self.w1 = torch.nn.Conv2d(4*chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n,  gene_size, 1, bias=False)
        self.dropout = torch.nn.Dropout2d(p=0.4)
        self.dropout2 = torch.nn.Dropout2d(p=0.4)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size




    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...]
        y = self.perc(x)
        y = torch.cat((y,x), dim=1)
        y = self.attent(y)
        y = self.dropout(y)
        # y = torch.cat((y,gene), dim=1 )
        y = self.w1(y)
        y = self.attent2(y)
        y = torch.nn.functional.relu(y)
        y = self.dropout2(y)
        y = self.w2(y)
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1

        gene = gene + y  * update_mask #pre_life_mask

        x = x[:, :x.shape[1] - self.gene_size, ...]
        x = torch.cat((x, gene), dim=1)



        return x


class EngramNCA_v4(torch.nn.Module):
    def __init__(self, chn=12, hidden_n_pre=128,hidden_n_prop = 128, gene_size=3):
        super().__init__()
        self.pre = MixedCA(chn, hidden_n_pre, gene_size)
        self.prop = MixedGenePropCA(chn, hidden_n_prop, gene_size)

    def forward(self, x, update_rate=0.5):
        x = self.pre(x, update_rate)
        x = self.prop(x, update_rate)
        return x



    
class LocalChannelAttention(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(LocalChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio


        self.fc1 = torch.nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False, padding=0)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)


        return x * out


class EngramNCA_v1(torch.nn.Module):
    def __init__(self, chn=12, hidden_n_pre=128, hidden_n_prop=128, gene_size=3):
        super().__init__()
        self.pre = GeneCA(chn, hidden_n_pre, gene_size)
        self.prop = GenePropCA(chn, hidden_n_prop, gene_size)

    def forward(self, x, update_rate=0.5):
        x = self.pre(x, update_rate)
        x = self.prop(x, update_rate)
        return x


class EngramNCA_v5(torch.nn.Module):
    def __init__(self, chn=12, hidden_n_pre=128, hidden_n_prop=128, gene_size=3):
        super().__init__()
        self.pre = MixedCA(chn, hidden_n_pre, gene_size)
        self.prop = MixedGenePropCA(chn, hidden_n_prop, gene_size)

    def forward(self, x, update_rate=0.5):
        x = self.pre(x, update_rate)
        x = self.prop(x, update_rate)
        return x


class EngramNCA_v3(torch.nn.Module):
    def __init__(self, chn=12, hidden_n_pre=128, hidden_n_prop=128, gene_size=3):
        super().__init__()
        self.pre = MixedCA(chn, hidden_n_pre, gene_size)
        self.prop = MixedGenePropCA(chn, hidden_n_prop, gene_size)

    def forward(self, x, update_rate=0.5):
        x = self.pre(x, update_rate)
        x = self.prop(x, update_rate)
        return x



    
class CA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96, mask_n = 0):
    super().__init__()
    self.chn = chn
    self.perc = torch.nn.Conv2d(chn, 8 * chn, 3, padding=1, padding_mode="zeros", bias=False)
    self.dropout = torch.nn.Dropout2d(p=0.4)
    self.dropout2 = torch.nn.Dropout2d(p=0.4)
    self.w1 = torch.nn.Conv2d(9*chn, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    #self.w2.weight.data.zero_()
    self.mask_n = mask_n

  def forward(self, x, update_rate=0.5):
      y = self.perc(x)

      y = torch.cat((y, x), dim=1)
      y = self.dropout(y)
      y = self.w1(y)
      y = torch.relu(y)
      y = self.dropout2(y)
      y = self.w2(y)
      b, c, h, w = y.shape
      update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
      px = torch.nn.functional.pad(x, [1,1,1,1])
      #pre_life_mask = torch.nn.functional.max_pool2d(px[:, None, 3, ...], 3, 1, ).cuda() > 0.1

      # Perform update
      x = x + y * update_mask #* pre_life_mask
      return x

  


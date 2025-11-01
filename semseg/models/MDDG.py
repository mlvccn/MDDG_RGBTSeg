import sys
sys.path.append('/data/tangchen/MDDG')
import torch
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import DynamicsHead
import time

class MDDG(BaseModel):
    def __init__(self, backbone: str = 'MDDG-B0', num_classes: int = 20, modals: list = ['img', 'aolp', 'dolp', 'nir'], Aux_loss: bool = False) -> None:
        super().__init__(backbone, num_classes, modals)
        self.Aux_loss = Aux_loss
        if Aux_loss:
            self.Aux_head1 = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
            self.Aux_head2 = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        self.decode_head = DynamicsHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 256, num_classes) 
        self.apply(self._init_weights)    

    def forward(self, x: list) -> list:
        
        if self.Aux_loss and self.training:  
            y, rgb, th ,raw_mask= self.backbone(x)
            y = self.decode_head(y)
            aux_pre1 = self.Aux_head1(rgb)
            aux_pre2 = self.Aux_head2(th)
            y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            aux_pre1 = F.interpolate(aux_pre1, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            aux_pre2 = F.interpolate(aux_pre2, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            return y, aux_pre1, aux_pre2
        else:
            y, _, _, raw_mask = self.backbone(x) # out_puts: fuse_feat, rgb_feat, th_feat  type: list
            y = self.decode_head(y)
            y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            return y, raw_mask

    def init_pretrained(self, pretrained: str = None) -> None:
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        msg = self.backbone.load_state_dict(checkpoint, strict=False)
        del checkpoint
        

if __name__ == '__main__':
    modals = ['img', 'thermal'] 
    num_minibatch = 1
    gpu=1
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision('high')
    model = MDDG('MDDG-B2', 9, modals)
    # model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    model = model.cuda(gpu)
    x = [torch.zeros(1, 3, 608, 800, dtype = torch.float32).cuda(gpu), torch.zeros(1, 3, 608, 800, dtype = torch.float32).cuda(gpu)]
    model.eval()
    # from fvcore.nn import flop_count_table, FlopCountAnalysis
    # from natten.flops import add_natten_handle
     # print(x[0].shape[0].dtype)
    # from natten.flops import get_flops
    # model = model.cuda(gpu)
    # x = [torch.zeros(1, 3, 480, 640, dtype = torch.float32).cuda(gpu), torch.zeros(1, 3, 480, 640, dtype = torch.float32).cuda(gpu)]
    # model.eval()
    # flop_ctr = get_flops(model, x)
    # # # flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    # print(flop_ctr)     
    from thop import profile
    total_ops, total_params  = profile(model, inputs=(x, ), verbose=False)
    print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    print("%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))

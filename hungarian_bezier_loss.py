import torch
import numpy as np
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from ..ddp_utils import is_dist_avail_and_initialized, get_world_size
from ..curve_utils import BezierSampler, cubic_bezier_curve_segment, get_valid_points
from ._utils import WeightedLoss
from .hungarian_loss import HungarianLoss
from .builder import LOSSES

query_num = 80 #50 for CULane, 40 for tusimple and llamas, 80 for curvelanes
max_lanes_per_images = 14 #4 for CULane and llamas, 5 for tusimple, 14 for curvelanes

# TODO: Speed-up Hungarian on GPU with tensors
class _HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    POTO matching, which maximizes the cost matrix.
    """

    def __init__(self, alpha=0.8, bezier_order=3, num_sample_points=100, k=7):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points, order=bezier_order)

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute the matrices for an entire batch (computation is all pairs, in a way includes the real loss function)
        # targets: each target: ['keypoints': L x N x 2]
        # B: batch size; Q: max lanes per-pred, G: total num ground-truth-lanes
        B, Q = outputs["logits"].shape
        target_keypoints = torch.cat([i['keypoints'] for i in targets], dim=0)  # G x N x 2
        target_sample_points = torch.cat([i['sample_points'] for i in targets], dim=0)  # G x num_sample_points x 2

        # Valid bezier segments
        target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
        target_sample_points = self.bezier_sampler.get_sample_points(target_keypoints)

        # target_valid_points = get_valid_points(target_sample_points)  # G x num_sample_points
        G, N = target_keypoints.shape[:2]
        out_prob = outputs["logits"].sigmoid()  # B x Q
        out_lane = outputs['curves']  # B x Q x N x 2
        sizes = [target['keypoints'].shape[0] for target in targets]

        # 1. Local maxima prior
        _, max_indices = torch.nn.functional.max_pool1d(out_prob.unsqueeze(1),
                                                        kernel_size=self.k, stride=1,
                                                        padding=(self.k - 1) // 2, return_indices=True)
        max_indices = max_indices.squeeze(1)  # B x Q
        indices = torch.arange(0, Q, dtype=out_prob.dtype, device=out_prob.device).unsqueeze(0).expand_as(max_indices)
        local_maxima = (max_indices == indices).flatten().unsqueeze(-1).expand(-1, G)  # BQ x G

        # Safe reshape
        out_prob = out_prob.flatten()  # BQ
        out_lane = out_lane.flatten(end_dim=1)  # BQ x N x 2

        # 2. Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        # For binary classification, it is just prob (understand this prob as objectiveness in OD)
        cost_label = out_prob.unsqueeze(-1).expand(-1, G)  # BQ x G

        # 3. Compute the curve sampling cost
        cost_curve = 1 - torch.cdist(self.bezier_sampler.get_sample_points(out_lane).flatten(start_dim=-2),#[1000,200]
                                     target_sample_points.flatten(start_dim=-2),#[57,200]
                                     p=1) / self.num_sample_points  # BQ x G [1000, 57]
        # Bound the cost to [0, 1]
        cost_curve = cost_curve.clamp(min=0, max=1)

        # Final cost matrix (scipy uses min instead of max)
        C = local_maxima * cost_label ** (1 - self.alpha) * cost_curve ** self.alpha
        C = -C.view(B, Q, -1).cpu()
        # Hungarian (weighted) on each image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # Return (pred_indices, target_indices) for each image
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    
class reverseMatcher(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bezier_sampler = BezierSampler(num_sample_points=100, order=3)

    @torch.no_grad()
    def forward(self, outputs, targets,valid_gt_num):
        
        # Compute the cost matrix for an entire batch, only matrices on diagonal should be used
        B = valid_gt_num # valid img num
        Q = max_lanes_per_images # of gt
        target_sample_points = targets

        out_lane = outputs  # B x Q x num_sample_points x 2
        sizes = np.array([query_num]).repeat(valid_gt_num).tolist()

        # Safe reshape
        out_lane = out_lane.flatten(end_dim=1)  # BQ x num_sample_points x 2
        target_sample_points = target_sample_points.flatten(end_dim=1)

        # - cost matrix
        C = 1-torch.cdist(self.bezier_sampler.get_sample_points(out_lane).flatten(start_dim=-2),
                                     self.bezier_sampler.get_sample_points(target_sample_points).flatten(start_dim=-2),
                                     p=1)/100              # BQ x G [72,900]

        # Final cost matrix (scipy uses min instead of max)
        C = -C.view(B, Q, -1).cpu()
        
        indices = []
        re_pred_mask_whole = torch.tensor([])
        # rematch(each query get a closest gt) with topk
        for i,c in enumerate(C.split(sizes, -1)):
            c = c[i] # only matrices on diagonal should be used
            
            # find topk matched  for each gt
            topk = int(query_num*4/25)
            _, idx1 = torch.sort(c)
            _, idx2 = torch.sort(idx1)
            topk_mask = torch.zeros_like(c)
            topk_mask[idx2<topk] = 1
            
            #rematch
            rematch_mask = torch.zeros_like(c)
            re_gt_idx = torch.argmin(c,dim=0)
            for i,j in enumerate(re_gt_idx):
                rematch_mask[j,i] = 1
                
            #rematch with topk
            re_pred_mask = topk_mask * rematch_mask
            re_pred_mask = torch.sum(re_pred_mask,dim=0)
            re_pred_idx = torch.arange(query_num)
            
            re_pred_mask_whole = torch.cat((re_pred_mask_whole,re_pred_mask))

            indices.append((re_gt_idx,re_pred_idx))

        return [(torch.as_tensor(i, dtype=torch.int64
                ), torch.as_tensor(j, dtype=torch.int64
                )) for i, j in indices],[(torch.as_tensor(j, dtype=torch.int64
                ), torch.as_tensor(i, dtype=torch.int64)) for i, j in indices],re_pred_mask_whole      


@LOSSES.register()
class HungarianBezierLoss(WeightedLoss):
    def __init__(self, curve_weight=1, label_weight=0.1, seg_weight=0.75, alpha=0.8,
                 num_sample_points=100, bezier_order=3, weight=None, size_average=None, reduce=None, reduction='mean',
                 ignore_index=-100, weight_seg=None, k=9):
        super().__init__(weight, size_average, reduce, reduction)
        self.curve_weight = 1  # Weight for sampled points' L1 distance error between curves
        self.div_weight = 0.0001
        self.shape_weight = 0.0005
        self.loc_weight = 0.2
        self.label_weight = 0.1  # Weight for classification error
        self.seg_weight = seg_weight  # Weight for binary segmentation auxiliary task
        self.weight_seg = weight_seg  # BCE loss weight
        self.ignore_index = ignore_index
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points, order=bezier_order)
        self.matcher = _HungarianMatcher(alpha=alpha, num_sample_points=num_sample_points, bezier_order=bezier_order,
                                         k=k)
        self.reversemather = reverseMatcher()
        if self.weight is not None and not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight).cuda()
        if self.weight_seg is not None and not isinstance(self.weight_seg, torch.Tensor):
            self.weight_seg = torch.tensor(self.weight_seg).cuda()
        self.register_buffer('pos_weight', self.weight[1] / self.weight[0])
        self.register_buffer('pos_weight_seg', self.weight_seg[1] / self.weight_seg[0])
    
        
    def forward(self, inputs, targets, net):
        outputs = net(inputs)
        output_curves = outputs['curves']
        
        # Jshape for Availability Constraint
        shape_loss = self.shapeloss(self.bezier_sampler.get_sample_points(output_curves))
        
        target_labels = torch.zeros_like(outputs['logits'])
        target_segmentations = torch.stack([target['segmentation_mask'] for target in targets])
        loss_loc = 0
        loss_div = 0
        total_targets = 0
        for i in targets:
            total_targets += i['keypoints'].numel()

        # CULane actually can produce a whole batch of no-lane images,
        # in which case, we just calculate the classification loss
        if total_targets > 0:
          
            #Remove the empty imgs in the batch, and make up gt to have 'max_lanes_per_images' lines
            valid_gt = None
            valid_gt_mask = torch.tensor([True],device=output_curves.device)
            valid_gt_mask = valid_gt_mask.repeat([len(targets)])
            valid_gt_num = len(targets)
            flag = 0
            for i in range(len(targets)):
                size = targets[i]['keypoints'].size()
                
                #Remove the empty imgs in the batch
                if size[0] == 0:
                    valid_gt_mask[i]=False
                    valid_gt_num = valid_gt_num-1
                    continue
                
                #make up gt to have 'max_lanes_per_images' lines
                gt_per_img = targets[i]['keypoints']
                filler = gt_per_img[0,:,:].unsqueeze(0).repeat([max_lanes_per_images-size[0],1,1])
                gt_per_img = torch.cat((gt_per_img,filler),dim=0)
                gt_per_img = gt_per_img.unsqueeze(0) #1,max_lanes_per_images,4,2
                
                # The first time, direct assignment
                if flag == 0: 
                    valid_gt = gt_per_img
                    flag = flag+1
                    continue
                
                valid_gt = torch.cat((valid_gt,gt_per_img),dim=0) #valid_gt_num,max_lanes_per_images,4,2

            out_curves_for_valid_gt = output_curves[valid_gt_mask] #valid_gt_num,query_num,4,2
            
            ind1,ind2,re_pred_mask_whole = self.reversemather(outputs=valid_gt, targets=out_curves_for_valid_gt,valid_gt_num=valid_gt_num)
            id1 = HungarianLoss.get_src_permutation_idx(ind1) #idx for gt
            id2 = HungarianLoss.get_src_permutation_idx(ind2) #idx for pred
            
            #Jloc for Availability Constraint
            reverse_gt = valid_gt[id1]
            out_curves_for_valid_gt = out_curves_for_valid_gt[id2]
            topk_re_mask = re_pred_mask_whole.bool()
            loss_loc = F.l1_loss(out_curves_for_valid_gt[topk_re_mask,0,:],reverse_gt[topk_re_mask,0,:]) 
            loss_loc += F.l1_loss(out_curves_for_valid_gt[topk_re_mask,-1,:],reverse_gt[topk_re_mask,-1,:])

            out_curves_for_valid_gt = self.bezier_sampler.get_sample_points(out_curves_for_valid_gt)
            valid = max_lanes_per_images # In case some gt(especially filler) is not matched by any pred
            gt_idx = id1[1] #real idx for gt, id1[0] is batch_idx 
            
            #compute loss_div_per_img
            for i in range(valid_gt_num):
                loss_div_per_img = 0
                div_mask_per_lanes = self.shapere(gt_idx,i,re_pred_mask_whole)
                for div_mask_per_gtlane in div_mask_per_lanes:
                    if out_curves_for_valid_gt[div_mask_per_gtlane].numel()!=0:
                        loss_div_per_img += self.divloss(out_curves_for_valid_gt[div_mask_per_gtlane].unsqueeze(0))
                    else:
                        valid = valid-1
                if valid!=0:
                    loss_div_per_img = loss_div_per_img/valid

                loss_div += loss_div_per_img
                valid = max_lanes_per_images
                
            #Take the average loss_div of a batch
            loss_div = loss_div/valid_gt_num
            

            indices = self.matcher(outputs=outputs, targets=targets)
            idx = HungarianLoss.get_src_permutation_idx(indices)
            output_curves = output_curves[idx]
            # Targets (rearrange each lane in the whole batch)
            # B x N x ... -> BN x ...
            target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_sample_points = torch.cat([t['sample_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            # Valid bezier segments
            target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
            target_sample_points = self.bezier_sampler.get_sample_points(target_keypoints)
            output_sample_points = self.bezier_sampler.get_sample_points(output_curves)
            
            target_labels[idx] = self.softlabel(output_sample_points,target_sample_points)   

        else:
            # For DDP
            target_sample_points = torch.tensor([], dtype=torch.float32, device=output_curves.device)

                
        # Loss
        loss_curve = self.point_loss(output_sample_points,target_sample_points)
        loss_label = self.classification_loss(inputs=outputs['logits'], targets=target_labels)
        loss_seg   = self.binary_seg_loss(inputs=outputs['segmentations'], targets=target_segmentations)
        loss = self.label_weight*loss_label+self.curve_weight*loss_curve+self.seg_weight*loss_seg+shape_loss*self.shape_weight+loss_loc*self.loc_weight+self.div_weight*loss_div
        return loss, {'loss dis': loss_label, 'loss curve': loss_curve,
                      'loss loc':loss_loc,
                      'loss div':loss_div,
                      'loss shape':shape_loss}

    def point_loss(self, inputs, targets, valid_points=None):
        # L1 loss on sample points
        # inputs/targets: Lines x N x 2
        # valid points: L x N

        if targets.numel() == 0:
            targets = inputs.clone().detach()
        loss = F.l1_loss(inputs, targets, reduction='none')
        if valid_points is not None:
            loss *= valid_points.unsqueeze(-1)
            normalizer = valid_points.sum()
        else:
            normalizer = targets.shape[0] * targets.shape[1]
            normalizer = torch.as_tensor([normalizer], dtype=inputs.dtype, device=inputs.device)
        if self.reduction == 'mean':
            if is_dist_avail_and_initialized():  # Global normalizer should be same across devices
                torch.distributed.all_reduce(normalizer)
            normalizer = torch.clamp(normalizer / get_world_size(), min=1).item()
            loss = loss.sum() / normalizer
        elif self.reduction == 'sum':  # Usually not needed, but let's have it anyway
            loss = loss.sum()

        return loss

    def classification_loss(self, inputs, targets):
        # Typical classification loss (cross entropy)
        # No need for permutation, assume target is matched to inputs

        # Negative weight as positive weight
        return F.binary_cross_entropy_with_logits(inputs.unsqueeze(1), targets.unsqueeze(1), pos_weight=self.pos_weight,
                                                  reduction=self.reduction) / self.pos_weight

    def binary_seg_loss(self, inputs, targets):
        # BCE segmentation loss with weighting and ignore index
        # No relation whatever to matching

        # Process inputs
        inputs = torch.nn.functional.interpolate(inputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
        inputs = inputs.squeeze(1)

        # Process targets
        valid_map = (targets != self.ignore_index)
        targets[~valid_map] = 0
        targets = targets.float()

        # Negative weight as positive weight
        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight_seg,
                                                  reduction='none') / self.pos_weight_seg
        loss *= valid_map

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
    #Diversity Constraint
    def divloss(self,output_curves):
        straight_len = (output_curves[:,:,-1,:]-output_curves[:,:,0,:]).pow(2)
        straight_len = torch.sum(straight_len,dim=-1)
        straight_len = straight_len.sqrt()
        curve_len = (output_curves[:,:,1:,:] - output_curves[:,:,:-1,:]).pow(2)
        curve_len = torch.sum(curve_len,dim=-1)
        curve_len = curve_len.sqrt()
        curve_len = torch.sum(curve_len,dim=-1)
        curvature = curve_len/straight_len
        curvature = torch.unsqueeze(curvature, dim=-1)
        loss_shape = torch.cdist(curvature,curvature,p=1)

        return -loss_shape.mean()
    
    # Jshape for Availability Constraint
    def shapeloss(self,output_curves):
        straight = (output_curves[:,:,-1,:]-output_curves[:,:,0,:]).pow(2)
        straight = torch.sum(straight,dim=-1)
        straight = straight.sqrt()
        curve_len = (output_curves[:,:,1:,:] - output_curves[:,:,:-1,:]).pow(2)
        curve_len = torch.sum(curve_len,dim=-1)
        curve_len = curve_len.sqrt()
        curve_len = torch.sum(curve_len,dim=-1)
        curvature = curve_len/straight

        return curvature.mean()
    
    # Discrimination Constraint
    def softlabel(self,output_sample_points,target_sample_points):
        loss_reg = (output_sample_points-target_sample_points).abs()
        loss_reg = torch.sum(loss_reg,dim=-1)/2
        loss_reg = torch.sum(loss_reg,dim=-1)/10
        softlabel = (-loss_reg).exp()
        softlabel = 0.5+0.5*softlabel
        return softlabel
    
    def shapere(self,gt_idx,i,re_pred_mask_whole):
        a1 = (1-re_pred_mask_whole).bool() #reverse True and False
        div_mask_per_lanes = []
        for i in range(max_lanes_per_images):  
            div_mask_per_lane = gt_idx==i
            #Remove querys from other images
            div_mask_per_lane[:i*query_num] = False
            div_mask_per_lane[i*query_num+query_num:] = False
            
            #Apply the upper limit value of the diversity constraint
            div_mask_per_lane[a1] = False
            
            div_mask_per_lanes.append(div_mask_per_lane)
        return div_mask_per_lanes

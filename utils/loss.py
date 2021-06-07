# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from shapely.geometry import Polygon
import shapely
import numpy as np


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def Cross_Entropy_Loss(input, target):
    Cross = nn.CrossEntropyLoss()
    loss = Cross(input, target)
    return loss


def Smooth_L1_loss(input, targrt):
    smooth = torch.nn.SmoothL1Loss()
    loss = smooth(input, targrt)
    return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.device = device
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def reorganize_targets(self, target, num_target):
        '''
        reorder the four coordinateds
        :param target:
        :param num_target:
        :return: sorted_target
        '''
        target = target.float()
        cls_of_target = target[:, 0]
        target = target[:, 1:9].view(num_target, 4, 2)

        x = target[..., 0]
        y = target[..., 1]
        y_sorted, y_indices = torch.sort(y)

        x_sorted = torch.zeros(num_target, 4).to(self.device)
        for i in range(0, num_target):
            x_sorted[i] = x[i, y_indices[i]]

        x_sorted[:, :2], x_top_indices = torch.sort(x_sorted[:, :2])
        x_sorted[:, 2:4], x_bottom_indices = torch.sort(x_sorted[:, 2:4], descending=True)
        for i in range(0, num_target):
            y_sorted[i, :2] = y_sorted[i, :2][x_top_indices[i]]
            y_sorted[i, 2:4] = y_sorted[i, 2:4][x_bottom_indices[i]]

        sorted_target = torch.zeros_like(torch.cat((x_sorted, y_sorted), dim=1))
        sorted_target[:, 0::2] = x_sorted
        sorted_target[:, 1::2] = y_sorted

        return torch.cat((cls_of_target.unsqueeze(1), sorted_target), 1)

    def loss_layer(self, output, target, stride, anchors):
        batch_size, _, output_size = output.shape[0:3]  # batch_size和yolo输出的大小
        num_anchors = len(anchors)
        device = self.device

        anchors = anchors.to(device)

        output_xy = output[..., 0:8]
        output_conf = output[..., 8]
        output_cls = output[..., 9:]

        t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls = \
            self.build_targets_seg(target, anchors, num_anchors, self.nc, output_size)
        tcls = tcls[mask]
        t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls = \
            t1_x.to(device), t1_y.to(device), t2_x.to(device), t2_y.to(device), t3_x.to(device), t3_y.to(
                device), t4_x.to(
                device), t4_y.to(device), mask.to(device), tcls.to(device)
        num_pred = mask.sum().float()  # Number of anchors (assigned to targets)

        k = num_pred / batch_size

        if num_pred > 0:
            lx1 = (k) * Smooth_L1_loss(output_xy[..., 0][mask], t1_x[mask]) / 8
            ly1 = (k) * Smooth_L1_loss(output_xy[..., 1][mask], t1_y[mask]) / 8
            lx2 = (k) * Smooth_L1_loss(output_xy[..., 2][mask], t2_x[mask]) / 8
            ly2 = (k) * Smooth_L1_loss(output_xy[..., 3][mask], t2_y[mask]) / 8
            lx3 = (k) * Smooth_L1_loss(output_xy[..., 4][mask], t3_x[mask]) / 8
            ly3 = (k) * Smooth_L1_loss(output_xy[..., 5][mask], t3_y[mask]) / 8
            lx4 = (k) * Smooth_L1_loss(output_xy[..., 6][mask], t4_x[mask]) / 8
            ly4 = (k) * Smooth_L1_loss(output_xy[..., 7][mask], t4_y[mask]) / 8

            conf_loss = (k * 10) * self.BCEobj(output_conf, mask.float())
            cls_loss = (k / self.nc) * Cross_Entropy_Loss(output_cls[mask], torch.argmax(tcls, 1))
        else:
            lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, conf_loss, cls_loss = \
                [torch.FloatTensor([0]).requires_grad_(True).to(device) for _ in range(10)]

        loc_loss = lx1 + ly1 + lx2 + ly2 + lx3 + ly3 + lx4 + ly4

        loss = loc_loss + conf_loss + cls_loss

        return loss, loc_loss, conf_loss, cls_loss

    def __call__(self, pred, targets, segment):
        device = targets.device

        idx, segment = segment.split((1, 9), 1)
        segment_targets = []
        max_idx = idx.max(0)[0]

        for i in range(int(max_idx) + 1):
            matched = (idx == i).squeeze()
            s = segment[matched]
            segment_targets.append(s)

        loss_small = self.loss_layer(pred[0], segment_targets, self.stride[0], self.anchors[0])
        loss_middle = self.loss_layer(pred[1], segment_targets, self.stride[1], self.anchors[1])
        loss_big = self.loss_layer(pred[2], segment_targets, self.stride[2], self.anchors[2])

        loss = loss_small[0] + loss_middle[0] + loss_big[0]
        loc_loss = loss_small[1] + loss_middle[1] + loss_big[1]
        conf_loss = loss_small[2] + loss_middle[2] + loss_big[2]
        cls_loss = loss_small[3] + loss_middle[3] + loss_big[3]

        loss = loss.reshape(1, )
        loc_loss = loc_loss.reshape(1, )
        conf_loss = conf_loss.reshape(1, )
        cls_loss = cls_loss.reshape(1, )

        return loss, torch.cat((loc_loss, conf_loss, cls_loss, loss)).detach()

    def call(self, pred, targets, segment):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(pred, targets, segment)  # targets

        # Losses
        for i, pi in enumerate(pred):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, pred, targets, segment):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_anch, num_tgt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, t_seg = [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        gain_seg = torch.ones(11, device=segment.device)
        ai = torch.arange(num_anch, device=targets.device).float().view(num_anch, 1).repeat(1, num_tgt)
        # same as .repeat_interleave(num_tgt)
        targets = torch.cat((targets.repeat(num_anch, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        segment = torch.cat((segment.repeat(num_anch, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain_seg[2:10] = torch.tensor(pred[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2]]

            # Match targets to anchors
            t = targets * gain  # 将targets缩放到特征图大小
            s = segment * gain_seg
            if num_tgt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                s = s[j]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def build_targets_seg(self, total_target, anchor_wh, num_anchor, num_class, output_size):
        """
        returns target_per_image, nCorrect, tx, ty, tw, th, tconf, tcls
        """
        num_image = len(total_target)  # number of images in batch
        target_per_image = [len(x) for x in total_target]  # targets per batch
        # batch size (4), number of anchors (3), number of grid points (13)
        t = torch.zeros(num_image * 8, num_anchor, output_size, output_size)
        t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y = t.chunk(8)
        del t

        mask = torch.BoolTensor(num_image, num_anchor, output_size, output_size).fill_(0)
        tcls = torch.ByteTensor(num_image, num_anchor, output_size, output_size, num_class).fill_(0)
        target_category = torch.ShortTensor(num_image, max(target_per_image)).fill_(-1)  # target category
        device = self.device

        for img_idx in range(num_image):
            num_taget_cur_image = target_per_image[img_idx]  # number of targets per image
            if num_taget_cur_image == 0:
                continue
            target = total_target[img_idx].to(device)
            target = self.reorganize_targets(target, num_taget_cur_image)
            target_category[img_idx, :num_taget_cur_image] = target[:, 0].long()

            box1 = target[:, 1:9] * output_size

            # Convert to position relative to box
            gp1_x, gp1_y, gp2_x, gp2_y, gp3_x, gp3_y, gp4_x, gp4_y = box1.chunk(8, dim=1)

            # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
            c_box = torch.clamp(torch.round(box1).long(), min=0, max=output_size - 1)
            gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j = c_box.chunk(8, dim=1)

            # Get each target center
            gp_x = torch.cat((gp1_x, gp2_x, gp3_x, gp4_x), 1).view(-1, 4)
            gp_y = torch.cat((gp1_y, gp2_y, gp3_y, gp4_y), 1).view(-1, 4)

            # min(x) max(x) min(y) max(y) in each instance
            gp_x_min = torch.min(gp_x, 1)[0]
            gp_x_max = torch.max(gp_x, 1)[0]
            gp_y_min = torch.min(gp_y, 1)[0]
            gp_y_max = torch.max(gp_y, 1)[0]

            # Set target center in a certain cell
            gp_x_center = torch.round(torch.mean((torch.stack((gp_x_min, gp_x_max), dim=1)), dim=1))
            gp_y_center = torch.round(torch.mean((torch.stack((gp_y_min, gp_y_max), dim=1)), dim=1))

            x_min = torch.clamp((gp_x_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                                 - anchor_wh[:, 0].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)
            x_max = torch.clamp((gp_x_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                                 + anchor_wh[:, 0].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)
            y_min = torch.clamp((gp_y_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                                 - anchor_wh[:, 1].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)
            y_max = torch.clamp((gp_y_center.unsqueeze(1).repeat(1, num_anchor).view(-1, num_anchor, 1)
                                 + anchor_wh[:, 1].view(-1, num_anchor, 1) / 2), min=0, max=output_size - 1)

            top_left = torch.cat((x_min.view(-1, 1), y_min.view(-1, 1)), 1)
            top_right = torch.cat((x_max.view(-1, 1), y_min.view(-1, 1)), 1)
            bottom_right = torch.cat((x_max.view(-1, 1), y_max.view(-1, 1)), 1)
            bottom_left = torch.cat((x_min.view(-1, 1), y_max.view(-1, 1)), 1)
            # Get bounding boxes
            box2 = torch.cat((top_left, top_right, bottom_right, bottom_left), 1).view(-1, num_anchor, 8)

            iou_anch = torch.zeros(num_anchor, num_taget_cur_image, 1)
            for i in range(0, num_taget_cur_image):
                for j in range(0, num_anchor):
                    polygon1 = Polygon(box1[i, :].view(4, 2)).convex_hull
                    polygon2 = Polygon(box2[i, j, :].view(4, 2)).convex_hull
                    if polygon1.intersects(polygon2):
                        try:
                            inter_area = polygon1.intersection(polygon2).area
                            union_area = polygon1.union(polygon2).area
                            iou_anch[j, i] = inter_area / union_area
                        except shapely.geos.TopologicalError:
                            print('shapely.geos.TopologicalError occured, iou set to 0')

            iou_anch = iou_anch.squeeze(2)
            # Select best iou_pred and anchor
            iou_anch_best, matched_anchor_idx = iou_anch.max(0)  # best anchor [0-2] for each target
            matched_anchor_idx = matched_anchor_idx.to(device)

            # Select best unique target-anchor combinations
            if num_taget_cur_image > 1:
                iou_order = np.argsort(-iou_anch_best)  # best to worst
                # from largest iou to smallest iou
                # Unique anchor selection (slower but retains original order)

                u = torch.cat((gp1_i, gp1_j, gp2_i, gp2_j, gp3_i, gp3_j, gp4_i, gp4_j, matched_anchor_idx.unsqueeze(1)),
                              0).view(-1, num_taget_cur_image).cpu().numpy()

                _, first_unique = np.unique(u[:, iou_order], axis=1,
                                            return_index=True)  # first unique indices; each cell response to on target
                i = iou_order[first_unique]

                # best anchor must share significant commonality (iou) with target
                i = i[iou_anch_best[i] > 0.1]
                if len(i) == 0:
                    continue

                matched_anchor_idx, target = matched_anchor_idx[i], target[i]
                if len(target.shape) == 1:
                    target = target.view(1, 5)
            else:
                if iou_anch_best < 0.1:
                    continue
                i = 0

            target_class = target[:, 0].long()

            target[:, 1:9] = torch.clamp(target[:, 1:9], min=0, max=1)
            gp1_x, gp1_y, gp2_x, gp2_y, gp3_x, gp3_y, gp4_x, gp4_y = (target[:, 1:9] * output_size).chunk(8, dim=1)

            # Get target center
            gp_x = torch.cat((gp1_x, gp2_x, gp3_x, gp4_x), 1).view(-1, 4)
            gp_y = torch.cat((gp1_y, gp2_y, gp3_y, gp4_y), 1).view(-1, 4)

            gp_x_min = torch.min(gp_x, 1)[0]
            gp_x_max = torch.max(gp_x, 1)[0]
            gp_y_min = torch.min(gp_y, 1)[0]
            gp_y_max = torch.max(gp_y, 1)[0]

            gp_x_center = torch.round(torch.mean((torch.stack((gp_x_min, gp_x_max), dim=1)), dim=1)).long()
            gp_y_center = torch.round(torch.mean((torch.stack((gp_y_min, gp_y_max), dim=1)), dim=1)).long()

            # Coordinates
            t1_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp1_x.squeeze(
                1).cpu() - gp_x_center.float().cpu()
            t1_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp1_y.squeeze(
                1).cpu() - gp_y_center.float().cpu()
            t2_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp2_x.squeeze(
                1).cpu() - gp_x_center.float().cpu()
            t2_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp2_y.squeeze(
                1).cpu() - gp_y_center.float().cpu()
            t3_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp3_x.squeeze(
                1).cpu() - gp_x_center.float().cpu()
            t3_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp3_y.squeeze(
                1).cpu() - gp_y_center.float().cpu()
            t4_x[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp4_x.squeeze(
                1).cpu() - gp_x_center.float().cpu()
            t4_y[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = gp4_y.squeeze(
                1).cpu() - gp_y_center.float().cpu()

            # One-hot encoding of label
            tcls[img_idx, matched_anchor_idx, gp_y_center, gp_x_center, target_class] = 1
            mask[img_idx, matched_anchor_idx, gp_y_center, gp_x_center] = 1

        return t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls

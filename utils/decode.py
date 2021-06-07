import torch


def build_decode(output, anchors, strides, num_classes):
    assert anchors.shape[0] == 3
    assert strides.shape[0] == 3
    small_pred = decode(output[0], strides[0], anchors[0], num_classes)
    middle_pred = decode(output[1], strides[1], anchors[1], num_classes)
    big_pred = decode(output[2], strides[2], anchors[2], num_classes)

    return [small_pred, middle_pred, big_pred]


def decode(output, stride, anchors, num_classes):
    device = output.device
    batch_size, _, ny, nx = output.shape[0:4]
    grid_x = torch.arange(nx, device=device).repeat(ny, 1).view(
        [1, 1, ny, nx]).float()
    grid_y = torch.arange(ny, device=device).repeat(nx, 1).t().view(
        [1, 1, ny, nx]).float()
    P1_x = output[..., 0]  # Point1 x
    P1_y = output[..., 1]  # Point1 y
    P2_x = output[..., 2]  # Point2 x
    P2_y = output[..., 3]  # Point2 y
    P3_x = output[..., 4]  # Point3 x
    P3_y = output[..., 5]  # Point3 y
    P4_x = output[..., 6]  # Point4 x
    P4_y = output[..., 7]  # Point4 y

    pred_boxes = torch.FloatTensor(batch_size, 3, ny, nx, 8).to(device)
    pred_conf = output[..., 8]  # Conf
    pred_cls = output[..., 9:]  # Class
    pred_boxes[..., 0] = P1_x + grid_x
    pred_boxes[..., 1] = P1_y + grid_y
    pred_boxes[..., 2] = P2_x + grid_x
    pred_boxes[..., 3] = P2_y + grid_y
    pred_boxes[..., 4] = P3_x + grid_x
    pred_boxes[..., 5] = P3_y + grid_y
    pred_boxes[..., 6] = P4_x + grid_x
    pred_boxes[..., 7] = P4_y + grid_y

    pred = torch.cat((pred_boxes.view(batch_size, -1, 8) * stride,
                      torch.sigmoid(pred_conf.view(batch_size, -1, 1)), pred_cls.view(batch_size, -1, num_classes)),
                     -1)
    return pred

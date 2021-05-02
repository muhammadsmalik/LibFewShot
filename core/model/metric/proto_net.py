import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel

# https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
class ProtoLayer(nn.Module):
    def __init__(self, way_num, shot_num, query_num):
        super(ProtoLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

    def forward(self, query_feat, support_feat):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c
        query_feat = query_feat.view(t, self.way_num * self.query_num, c)
        # t, w, c
        support_feat = support_feat.view(t, self.way_num, self.shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)

        # t, wq, 1, c - t, 1, w, c -> t, wq, w
        dist = -torch.sum(torch.pow(query_feat.unsqueeze(2) -
                                    proto_feat.unsqueeze(1), 2), dim=3)

        return dist


class ProtoNet(MetricModel):
    def __init__(self, way_num, shot_num, query_num, emb_func, device):
        super(ProtoNet, self).__init__(way_num, shot_num, query_num, emb_func, device)
        self.proto_layer = ProtoLayer(way_num, shot_num, query_num)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat,mode=1)

        output = self.proto_layer(query_feat, support_feat) \
            .view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target)

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        emb = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(emb,mode=1)

        output = self.proto_layer(query_feat, support_feat) \
            .view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target)
        acc = accuracy(output, query_target)

        return output, acc, loss

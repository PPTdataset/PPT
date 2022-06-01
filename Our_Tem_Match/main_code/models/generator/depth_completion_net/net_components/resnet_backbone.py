import torch
from torchvision.models import resnet18, resnet34, resnet50


_MODEL_URLS = {
    'resnet18': '/test/Models/resnet18-5c106cde.pth',
    'resnet34': '/test/Models/resnet34-333f7ec4.pth',
    'resnet50': '/test/Models/resnet50-19c8e357.pth'
}


def resnet18_backbone(pre_trained=False, **kwargs):
    model = resnet18(pretrained=pre_trained, **kwargs)
    # if pre_trained:
    #     state_dict = torch.load(_MODEL_URLS['resnet18'])
    #     model.load_state_dict(state_dict, strict=False)
    return model


def resnet34_backbone(pre_trained=False, **kwargs):
    model = resnet34(pretrained=pre_trained, **kwargs)
    # if pre_trained:
    #     state_dict = torch.load(_MODEL_URLS['resnet34'])
    #     model.load_state_dict(state_dict, strict=False)
    return model


def resnet50_backbone(opt=None, pre_trained=False, **kwargs):
    model = resnet50(pretrained=False, **kwargs)
    if pre_trained:
        print("pre_trained")
        state_dict = torch.load("%s/temp_data/pretrained_model/resnet50-19c8e357.pth" %opt.ROOT)
        model.load_state_dict(state_dict)
    return model

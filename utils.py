import os
import torch
import numpy as np

def save_model(model, name, folder_name):
    # torch.save(model, (os.path.join(folder_name, "trained_" + name + ".pth")))
    torch.save(model.state_dict(), (os.path.join(folder_name, "trained_" + name + ".pth")))
    # print(model.state_dict().keys())
    # print("Done saving Model")

def load_model(model, path):
    '''
    model = torch.load(path)
    print(model.state_dict().keys())
    '''
    checkpoint = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    flag, flag2 = 0, 0
    for k, v in checkpoint.items():
        if '_estimated' in k:
            model.register_buffer(k, v)
            flag2 = 1
        elif k not in model_dict:
            # print("checkpoint", k, v.size())
            if k == 'rel_attention':
                flag = 1
    if flag:
        checkpoint['rel_attention.rel_attention'] = checkpoint['rel_attention']
    if flag2:
        print("loading ewc params")
    for k, v in model_dict.items():
        if k not in checkpoint:
            print("model_dict", k, v.size())
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)


    return model



import torch


def ckeck_binary(param):
    # pick up the fisrt element to infer the alpha
    alpha = torch.abs(param[0,0,0,0])
    return alpha, torch.equal(param.sign(), param/alpha)

def main():
    checkpoint = torch.load("ResNet18-57.30.pth.tar")
    pretrained_model = checkpoint['state_dict']

    for name in pretrained_model.keys():
        if 'conv' in name and 'conv1' not in name: # skip the first conv layer
            alpha, is_binary = ckeck_binary(pretrained_model[name])
            print("Is the {} binary? -> {} \t AND the alpha is {:0.6f}".format(name, is_binary, alpha))

if __name__ == '__main__':
    main()
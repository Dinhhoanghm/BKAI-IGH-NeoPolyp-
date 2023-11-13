import argparse
from ..model.model import NeoPolypModel
from ..dataset.dataset import NeoPolypDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
import torch
import os

parser = argparse.ArgumentParser(description='NeoPolyp Inference')
parser.add_argument('--model', type=str, default='model.pth',
                    help='model path')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers')
parser.add_argument('--save_path', type=str, default='/kaggle/working/predicted_masks',
                    help='save path')
args = parser.parse_args()


def main():
    model = NeoPolypModel.load_from_checkpoint(args.model)
    model.eval()
    test_dataset = NeoPolypDataset("test", args.data_path)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    if not os.path.isdir(args.save_path):
        os.mkdir(args.sav)
    for _, (img, file_id, H, W) in enumerate(test_dataloader):
        with torch.no_grad():
            predicted_mask = model(img)
        for i in range(args.batch_size):
            filename = file_id + ".png"
            one_hot = F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()
            mask2img = Resize((H[i].item(), W[i].item()), interpolation=InterpolationMode.NEAREST)(
                ToPILImage()(one_hot))
            mask2img.save(os.path.join(args.save_path, filename))


if __name__ == '__main__':
    main()

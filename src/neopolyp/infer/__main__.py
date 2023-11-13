import argparse
from ..model.model import NeoPolypModel
from ..dataset.dataset import NeoPolypDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
import torch
import os
import pandas as pd
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='NeoPolyp Inference')
parser.add_argument('--model', type=str, default='model.pth',
                    help='model path')
parser.add_argument('--data_path', type=str, default='data',
                    help='data path')
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
        os.mkdir(args.save_path)
    for _, (img, file_id, H, W) in enumerate(test_dataloader):
        with torch.no_grad():
            predicted_mask = model(img.cuda())
        for i in range(args.batch_size):
            filename = file_id[i] + ".png"
            one_hot = F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()
            mask2img = Resize((H[i].item(), W[i].item()), interpolation=InterpolationMode.NEAREST)(
                ToPILImage()(one_hot))
            mask2img.save(os.path.join(args.save_path, filename))

    def rle_to_string(runs):
        return ' '.join(str(x) for x in runs)

    def rle_encode_one_mask(mask):
        pixels = mask.flatten()
        pixels[pixels > 0] = 255
        use_padding = False
        if pixels[0] or pixels[-1]:
            use_padding = True
            pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
            pixel_padded[1:-1] = pixels
            pixels = pixel_padded

        rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
        if use_padding:
            rle = rle - 1
        rle[1::2] = rle[1::2] - rle[:-1:2]
        return rle_to_string(rle)

    def mask2string(dir):
        # mask --> string
        strings = []
        ids = []
        ws, hs = [[] for i in range(2)]
        for image_id in os.listdir(dir):
            id = image_id.split('.')[0]
            path = os.path.join(dir, image_id)
            print(path)
            img = cv2.imread(path)[:, :, ::-1]
            h, w = img.shape[0], img.shape[1]
            for channel in range(2):
                ws.append(w)
                hs.append(h)
                ids.append(f'{id}_{channel}')
                string = rle_encode_one_mask(img[:, :, channel])
                strings.append(string)
        r = {
            'ids': ids,
            'strings': strings,
        }
        return r

    res = mask2string(args.save_path)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    df.to_csv(r'output.csv', index=False)


if __name__ == '__main__':
    main()

import os
import argparse
import torch
import torch.utils.data as data
from torch.distributed.elastic.multiprocessing.errors import record

from milwsi.datasets import build_dataset
from milwsi.model.backbone import resnet50_baseline
from milwsi.utils import init_dist, get_dist_info


def parse_options():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', default='WSIDataset')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--feature_path', type=str, default='')
    args = parser.parse_args()
    return args


@record
def main():
    args = parse_options()

    # distributed settings
    if args.launcher == 'pytorch':
        distributed = True
        init_dist()
        device = torch.cuda.current_device()
    else:
        distributed = False
        device = torch.device(args.device)

    local_rank, world_size = get_dist_info()

    # create output folder
    os.makedirs(args.feature_path, exist_ok=True)

    # build model and dataset
    model = resnet50_baseline(pretrained=True)
    model = model.to(device)
    model.eval()
    dataset = build_dataset(name=args.dataset, path=args.data_path)

    if distributed:
        sampler = data.DistributedSampler(dataset)
        loader = data.DataLoader(dataset=dataset, batch_size=1, sampler=sampler, collate_fn=lambda x: x[0])
    else:
        loader = data.DataLoader(dataset=dataset, batch_size=1, collate_fn=lambda x: x[0])

    total = len(loader)

    for idx, batch in enumerate(loader):
        wsi_id = batch['wsi_id']
        wsi_path = batch['wsi_path']
        save_path = os.path.join(args.feature_path, wsi_id + '.pt')

        print(f"rank {local_rank} / {world_size}, progress: {idx + 1}/{total}, wsi_id: {wsi_id}")
        if os.path.exists(save_path):
            continue

        bag = build_dataset(name="WSIBag", paths=wsi_path)
        bag_loader = data.DataLoader(dataset=bag, batch_size=args.batch_size, num_workers=4)

        # extract feature and save .pt file.
        for bag_data in bag_loader:
            with torch.no_grad():
                bag_data = bag_data.to(device)
                features = model(bag_data)
                torch.save(features, save_path)


if __name__ == '__main__':
    main()

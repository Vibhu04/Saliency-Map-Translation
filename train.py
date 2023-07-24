import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import torch
import torch.optim as optim
import yaml
import argparse
from torch.utils.data import DataLoader
from transformers import AutoProcessor, BlipModel
from torch.utils.tensorboard import SummaryWriter


from vg_dataloader import VG_Dataset
from ViT import RegionEncoder


def main():

    parser = arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img_dir = args.img_dir
    annot_file = args.annot_file

    blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    dataset = VG_Dataset(img_dir, annot_file)

    batch_size = args.batch_size
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder_model = RegionEncoder(args.d_model, args.nhead, args.num_encoder_layers)

    encoder_model = encoder_model.to(device)

    optimizer = optim.Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if os.path.exists(args.checkpoint_save_name):
        checkpoint = torch.load(args.checkpoint_save_name, map_location=device)
        encoder_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Existing model loaded")
    else:
        print("No existing model found")

    if args.write_logs:
        writer = SummaryWriter(args.logs_dir)
    else:
        writer = None


    train(encoder_model, train_dataloader, blip, processor, optimizer, args.num_epochs, writer, device, args)




def train(encoder_model, dataloader, blip, processor, optimizer, num_epochs, writer, device, args):

    for epoch in range(num_epochs):
        encoder_model.train()
        blip.eval()
        running_loss = 0.0

        for batch_idx, (images, regions, captions) in enumerate(dataloader):

            images, regions = images.to(device), regions.to(device)
            optimizer.zero_grad()
            encodings = encoder_model(images, regions, device)
            encodings = encodings.reshape(-1, 64, 64, 3)

            inputs = processor(text=captions, images=encodings, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = blip(**inputs)

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)
            probs = 1 - probs

            probs.backward()
            optimizer.step()

            running_loss += probs.item() * inputs.size(0)

            if writer is None:
                writer.add_scalar("train_loss", float(probs.detach().data), batch_idx + 1 + args.batch_size * epoch)


        # Compute average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")




def arg_parser():

    with open('config.yml', 'r') as file:
        args = yaml.safe_load(file)

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, default=args['img_dir'], help='Images directory')
    parser.add_argument('--annot_file', type=str, default=args['annot_file'], help='Annotation file')
    parser.add_argument('--batch_size', type=int, default=args['batch_size'], help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=args['num_epochs'], help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=args['d_model'], help='Model inner dimension')
    parser.add_argument('--nhead', type=int, default=args['nhead'], help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=args['num_encoder_layers'],
                        help='Number of encoder layers')
    parser.add_argument('--lr', type=int, default=args['lr'], help='Learning rate')
    parser.add_argument('--weight_decay', type=int, default=args['weight_decay'], help='Weight decay coefficient')
    parser.add_argument('--checkpoint_save_name', type=str, default=args['checkpoint_save_name'],
                        help="File path of checkpoint")
    parser.add_argument('--write_logs', type=bool, default=args['write_logs'],
                        help="Bool for writing logs with tensorboard")
    parser.add_argument('--logs_dir', type=str, default=args['logs_dir'], help="Logs directory")


    return parser





if __name__ == '__main__':
    main()
    


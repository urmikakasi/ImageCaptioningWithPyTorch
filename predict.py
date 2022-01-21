import torch

from keras.preprocessing.image import load_img, img_to_array                    # To load image into memory and to convert image to numpy array
from keras.preprocessing.sequence import pad_sequences    
from transformers import BertTokenizer
from PIL import Image
import argparse
from IPython.display import Image as ImDisp
from models import caption
from datasets import coco, utils
from configuration import Config
import os
import numpy as np
import matplotlib.pyplot as plt                                                 # To display images
# %matplotlib inline

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

image = Image.open(image_path)
ImDisp(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0)



def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption



class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
    

# print(color.BOLD + 'Hello World !' + color.END)

output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
#result = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n\n\n"+color.BOLD+ result.capitalize()+ color.END)

def view_img_caption_pair():
  '''
  Given the model's path, image's path and tokenizer's path
  it generates a caption for the provided image and displays
  the same with the original image.
  '''
  text = result
  im  = np.array(load_img(image_path))
  fig, ax = plt.subplots(1, 2, figsize = (10,10))

  # Display the image
  ax[0].imshow(im)
  ax[0].axis('off')
  ax[0].set_title('Image', fontsize = 15)

  # Display caption beside the image
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  ax[1].set_xlim([0,0.5])
  ax[1].set_ylim([0,0.5])
  ax[1].text(0, 0.45, text, transform=ax[1].transAxes, fontsize=20,
        verticalalignment='center', bbox=props)
  ax[1].axis('off');

view_img_caption_pair()
 

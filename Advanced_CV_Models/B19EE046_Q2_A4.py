import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import PIL
from tqdm import tqdm
import cv2
from sklearn.metrics import classification_report, roc_curve
import pandas as pd
import re
import string
from sklearn.preprocessing import label_binarize

def transform_fl(img):
  img = img.resize((64,64))
  tensor = transforms.ToTensor()(img)
  return tensor

class CaptionVectorizer:
  def __init__(self, vocab, bos_token='<bos>', eos_token='<eos>', unk_token='<unk>'):
    self.vocab = vocab
    self.bos_token = bos_token
    self.eos_token = eos_token
    self.unk_token = unk_token
    self.bos_index = self.vocab.get(bos_token, len(self.vocab))
    if self.bos_index == len(self.vocab):
      self.vocab[bos_token] = self.bos_index
    self.eos_index = self.vocab.get(eos_token, len(self.vocab))
    if self.eos_index == len(self.vocab):
      self.vocab[eos_token] = self.eos_index
    self.unk_index = self.vocab.get(unk_token, len(self.vocab))
    if self.unk_index == len(self.vocab):
      self.vocab[unk_token] = self.unk_index

  def __call__(self, caption):
    caption = re.sub(r'http\S+', '', caption)
    caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
    caption = ''.join([i for i in caption if not i.isdigit()])
    words = caption.split()
    indices = [self.bos_index]
    for word in words:
      if word in self.vocab:
        indices.append(self.vocab[word])
      else:
        indices.append(self.unk_index)
    indices.append(self.eos_index)
    return torch.LongTensor(indices)

  def return_vocab(self):
    return self.vocab

  def get_token_from_index(self, index):
    for token, idx in self.vocab.items():
      if idx == index:
        return token
    return self.unk_token

class creator(Dataset):
  def __init__(self, img_dir, dataframe, caption_vectorizer):
    self.img_dir = img_dir 
    self.dataframe = dataframe
    self.caption_vectorizer = caption_vectorizer

  def __getitem__(self, idx):
    self.img_name = self.dataframe.iloc[idx]['images']
    self.img_caption = self.dataframe.iloc[idx]['captions']
    self.img_path = os.path.join(self.img_dir,self.img_name)
    self.img = cv2.imread(self.img_path)
    if self.img is not None:
      self.img = cv2.resize(self.img,(224,224))
      self.vector = self.caption_vectorizer(self.img_caption)
      return transforms.ToTensor()(self.img), self.vector

  def __len__(self):
    return self.dataframe.shape[0]

class EncoderDecoder(pl.LightningModule):
  def __init__(self, cnn_name, mode, hidden_dim, output_dim, vocab_size):
    '''
    input_dim = number of channels (input to CNN)
    vocab_size = size of vocabulary (input to RNN's embedding layer)
    output_dim = number of classes (within vocabulary)
    '''
    super().__init__()
    self.loss=[]
    self.encoder_name = cnn_name
    self.mode = mode
    if self.encoder_name=='resnet_34' and self.mode=='Train':
      self.encoder = models.resnet34(weights='IMAGENET1K_V1')
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features
    if self.encoder_name=='convnext_tiny' and self.mode=='Train':
      self.encoder = models.convnext_tiny(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features 
    if self.encoder_name=='mnasnet0_5' and self.mode=='Train':
      self.encoder = models.mnasnet0_5(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features 
    if self.encoder_name=='regnet_y_400mf' and self.mode=='Train':
      self.encoder = models.regnet_y_400mf(weights='IMAGENET1K_V1')
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='resnext50_32x4d' and self.mode=='Train':
      self.encoder = models.resnext50_32x4d(weights='IMAGENET1K_V1')
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='shufflenet_v2_x0_5' and self.mode=='Train':
      self.encoder = models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='squeezenet1_0' and self.mode=='Train':
      self.encoder = models.squeezenet1_0(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier[-3].in_channels
      self.out = self.encoder.classifier[-3].out_channels
    if self.encoder_name=='vgg11' and self.mode=='Train':
      self.encoder = models.vgg11(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features 
    if self.encoder_name=='wide_resnet50_2' and self.mode=='Train':
      self.encoder = models.wide_resnet50_2(weights='IMAGENET1K_V1')
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='swin_t' and self.mode=='Train':
      self.encoder = models.swin_t(weights='IMAGENET1K_V1')
      self.features = self.encoder.head.in_features
      self.out = self.encoder.head.out_features
    if self.encoder_name=='densenet_121' and self.mode=='Train':
      self.encoder = models.densenet121(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier.in_features
      self.out = self.encoder.classifier.out_features
    if self.encoder_name=='efficientnet_b0' and self.mode=='Train':
      self.encoder = models.efficientnet.efficientnet_b0(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features
    if self.encoder_name=='mobilenet_v3_large' and self.mode=='Train':
      self.encoder = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features
    if self.encoder_name=='resnet_34' and self.mode=='Test':
      self.encoder = models.resnet34(weights=None)
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features
    if self.encoder_name=='convnext_tiny' and self.mode=='Test':
      self.encoder = models.convnext_tiny(weights=None)
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features 
    if self.encoder_name=='mnasnet0_5' and self.mode=='Test':
      self.encoder = models.mnasnet0_5(weights=None)
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features 
    if self.encoder_name=='regnet_y_400mf' and self.mode=='Test':
      self.encoder = models.regnet_y_400mf(weights=None)
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='resnext50_32x4d' and self.mode=='Test':
      self.encoder = models.resnext50_32x4d(weights=None)
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='shufflenet_v2_x0_5' and self.mode=='Test':
      self.encoder = models.shufflenet_v2_x0_5(weights=None)
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='squeezenet1_0' and self.mode=='Test':
      self.encoder = models.squeezenet1_0(weights=None)
      self.features = self.encoder.classifier[-3].in_channels
      self.out = self.encoder.classifier[-3].out_channels
    if self.encoder_name=='vgg11' and self.mode=='Test':
      self.encoder = models.vgg11(weights=None)
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features 
    if self.encoder_name=='wide_resnet50_2' and self.mode=='Test':
      self.encoder = models.wide_resnet50_2(weights=None)
      self.features = self.encoder.fc.in_features
      self.out = self.encoder.fc.out_features 
    if self.encoder_name=='swin_t' and self.mode=='Test':
      self.encoder = models.swin_t(weights=None)
      self.features = self.encoder.head.in_features
      self.out = self.encoder.head.out_features
    if self.encoder_name=='densenet_121' and self.mode=='Test':
      self.encoder = models.densenet121(weights=None)
      self.features = self.encoder.classifier.in_features
      self.out = self.encoder.classifier.out_features
    if self.encoder_name=='efficientnet_b0' and self.mode=='Test':
      self.encoder = models.efficientnet.efficientnet_b0(weights=None)
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features
    if self.encoder_name=='mobilenet_v3_large' and self.mode=='Test':
      self.encoder = models.mobilenet_v3_large(weights=None)
      self.features = self.encoder.classifier[-1].in_features 
      self.out = self.encoder.classifier[-1].out_features
    for name, param in self.encoder.named_parameters():
        param.requires_grad = False
    self.embedding = nn.Embedding(vocab_size, self.out)
    self.decoder = nn.LSTM(2*self.out, hidden_dim, num_layers=1, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.vocab_size = vocab_size
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

  def forward(self, image, caption):
    caption = self.embedding(caption)
    image = self.encoder(image)
    image = image.unsqueeze(1).repeat(1, caption.size(1), 1)
    inputs = torch.cat((caption, image), dim=2)
    outputs, _ = self.decoder(inputs)
    outputs = self.fc(outputs)
    return outputs

  def training_step(self, batch, batch_idx):
    images, captions = batch
    outputs = self(images, captions[:,:-1])
    loss = self.loss_fn(outputs.reshape(outputs.shape[0]*outputs.shape[1], self.vocab_size-1), captions[:,1:].reshape(-1))
    if batch_idx==1874:
      print(loss)
    return loss

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

def create(images_file=None, token_file=None):
  token_files = []
  token_captions = []
  if token_file is not None:
    with open(token_file, 'r') as f:
      for line in f:
        image_name, caption_text = line.strip().split('\t')
        image_name = image_name.split('#')[0]
        token_files.append(image_name)
        token_captions.append(caption_text)
  with open(images_file, 'r') as f:
    image_list = f.read().splitlines()
  return image_list, token_files, token_captions

def return_dataframe(reference_list, imgs, captions):
  token_imgs = []
  token_captions = []
  for item in reference_list:
    for img_index in range(len(imgs)):
      if imgs[img_index]==item:
        token_imgs.append(item)
        token_captions.append(captions[img_index])
  df = pd.DataFrame()
  df['images']=token_imgs
  df['captions']=token_captions
  return df

def collate_fn(data):
  images, captions = zip(*data)
  captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
  return torch.stack(images), captions

def generate_caption(model, image_tensor, caption_vectorizer):
  model.eval().cuda()
  with torch.no_grad():
    features = model.encoder(image_tensor)
    features = features.unsqueeze(1)
    inputs = torch.LongTensor([[caption_vectorizer.bos_index]]).cuda()
    caption = []
    MAX_SEQ_LEN = 20
    for i in range(MAX_SEQ_LEN):
      caption.append(inputs.item())
      embeddings = model.embedding(inputs)
      inputs = torch.cat((features, embeddings), dim=2)
      outputs, hidden = model.decoder(inputs)
      predicted = outputs.argmax(-1)[:,-1]
      if predicted.item() == caption_vectorizer.eos_index:
        break
      inputs = predicted.unsqueeze(1)
  caption = [caption_vectorizer.get_token_from_index(w) for w in caption]
  return image_tensor.detach().cpu().numpy(), ' '.join(caption)

train_imgs, token_imgs , token_captions = create('/content/drive/MyDrive/CV_Assignment_4/Q2/Flickr_8k.trainImages.txt','/content/drive/MyDrive/CV_Assignment_4/Q2/Flickr8k.token.txt')
test_imgs, _, _ = create('/content/drive/MyDrive/CV_Assignment_4/Q2/Flickr_8k.testImages.txt',None)
df_train = return_dataframe(train_imgs, token_imgs, token_captions)
df_test = return_dataframe(test_imgs, token_imgs, token_captions)
df_train.to_csv('/content/drive/MyDrive/CV_Assignment_4/Q2/train_dataframe.csv')
df_test.to_csv('/content/drive/MyDrive/CV_Assignment_4/Q2/test_dataframe.csv')

df_train = pd.read_csv('/content/drive/MyDrive/CV_Assignment_4/Q2/train_dataframe.csv')
df_test = pd.read_csv('/content/drive/MyDrive/CV_Assignment_4/Q2/test_dataframe.csv')

df_train_list = df_train['captions'].to_list()
vocabulary={}
for caption in df_train_list:
  caption = re.sub(r'http\S+', '', caption)
  caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
  caption = ''.join([i for i in caption if not i.isdigit()])
  words = caption.split()
  for word in words:
    if word not in vocabulary:
      vocabulary[word] = len(vocabulary)
caption_vectorizer = CaptionVectorizer(vocabulary)
print(len(caption_vectorizer.return_vocab().keys()))
print(caption_vectorizer.return_vocab())

train_dataset = creator('/content/drive/MyDrive/CV_Assignment_4/Q2/Flicker8k_Dataset',df_train, caption_vectorizer)
test_dataset = creator('/content/drive/MyDrive/CV_Assignment_4/Q2/Flicker8k_Dataset',df_test, caption_vectorizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = EncoderDecoder(cnn_name='squeezenet1_0', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=10, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Models/squeezenet1_0/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

model = EncoderDecoder(cnn_name='resnet_34', hidden_dim=512, mode='Train', output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=50, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q2/Models/resnet_34/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

model = EncoderDecoder(cnn_name='swin_t', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=10, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q2/Models/swin_t/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

model = EncoderDecoder(cnn_name='mobilenet_v3_large', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=10, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q2/Models/mobilenet_v3_large/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

model = EncoderDecoder(cnn_name='densenet_121', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=10, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q2/Models/densenet_121/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

model = EncoderDecoder(cnn_name='vgg11', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=10, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q2/Models/vgg11/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

model = EncoderDecoder(cnn_name='efficientnet_b0', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab))
trainer = pl.Trainer(accelerator='gpu', precision=16, max_epochs=10, default_root_dir='/content/drive/MyDrive/CV_Assignment_4/Q2/Models/efficientnet_b0/', benchmark=True)
trainer.fit(model, train_loader)
del model, trainer

imgo = 0
for i,j in test_loader:
  imgo = i.cuda()
  break

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/squeezenet1_0/lightning_logs/version_0/checkpoints/epoch=9-step=18750.ckpt',EncoderDecoder(cnn_name='squeezenet1_0', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/densenet_121/lightning_logs/version_0/checkpoints/epoch=9-step=18750.ckpt',EncoderDecoder(cnn_name='densenet_121', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/efficientnet_b0/lightning_logs/version_0/checkpoints/epoch=9-step=18750.ckpt',EncoderDecoder(cnn_name='efficientnet_b0', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/mobilenet_v3_large/lightning_logs/version_1/checkpoints/epoch=9-step=18750.ckpt',EncoderDecoder(cnn_name='mobilenet_v3_large', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/resnet_34/lightning_logs/version_1/checkpoints/epoch=49-step=93750.ckpt',EncoderDecoder(cnn_name='resnet_34', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/swin_t/lightning_logs/version_0/checkpoints/epoch=9-step=18750.ckpt',EncoderDecoder(cnn_name='swin_t', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")

model = model_return('/content/drive/MyDrive/CV_Assignment_4/Q2/Models/vgg11/lightning_logs/version_0/checkpoints/epoch=9-step=18750.ckpt',EncoderDecoder(cnn_name='vgg11', mode='Test', hidden_dim=512, output_dim=caption_vectorizer.unk_index, vocab_size=len(caption_vectorizer.vocab)))
img, caption = generate_caption(model, imgo, caption_vectorizer)
print(caption)
cv2.imshow(img[0].transpose((1,2,0))*255,"Prompt")
from PIL import Image
from torchvision import transforms

def image_format(img, maximum_size=300, image_shape=None):
  image = Image.open(img).convert('RGB')
  if max(image.size)>maximum_size:
    size = maximum_size
  else:
    size = max(image.size)
  if image_shape is not None:
    size = image_shape
  
  image_shape_tranform =transforms.Compose([transforms.Resize(size),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
  
  image = image_shape_tranform(image)[:3,:,:].unsqueeze(0)
  return image
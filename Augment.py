from PIL import Image, ImageOps,ImageEnhance
import random
import torchvision
import torchvision.transforms as transforms

rotate_range = (-10, 10)
shift_range = (0.1, 0.1)  # Shift range as a fraction of the image size
scale_range = (0.9, 1.1)  # Zoom scale range
#Define data augmentation and further preprocessing steps
def random_color():
    return random.choice([(0), (128), (255)])
  
class RandomPadding:
    def __init__(self, min_pad=15, max_pad=45):
        self.min_pad = min_pad
        self.max_pad = max_pad
    def __call__(self, img):
        pad_left = random.randint(self.min_pad, self.max_pad)
        pad_top = random.randint(self.min_pad, self.max_pad)
        pad_right = random.randint(self.min_pad, self.max_pad)
        pad_bottom = random.randint(self.min_pad, self.max_pad)
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        pad_color = random_color()
        img = ImageOps.expand(img, padding, pad_color)
        return img

class RandomZoom:
    def __init__(self, zoom_range=(0.95, 1.2)):
        self.zoom_range = zoom_range
    def __call__(self, img):
        width, height = img.size
        zoom_factor = random.uniform(*self.zoom_range)
        new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
        img = img.resize((new_width, new_height), resample=Image.BICUBIC)
        if zoom_factor > 1:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))
        else:
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            img = ImageOps.expand(img, (left, top, left, top), fill=0)
        return img

class RandomInvert:
    def __call__(self, img):
        if random.random() < 0.5:
            img = ImageOps.invert(img)
        return img
      
data_transforms = transforms.Compose([
    RandomPadding(),
    RandomZoom(),
    transforms.Resize((300,300)),
    transforms.RandomAffine(degrees=rotate_range, translate=shift_range, scale=scale_range),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(),
    RandomInvert(),
    transforms.ToTensor()
])

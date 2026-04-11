import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TinyImageNet(Dataset):
    """Tiny ImageNet dataset.

    Args:
        root (string): Root directory of the dataset
        train (bool, optional): If True, creates the training dataset, otherwise creates the test dataset
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version
        download (bool, optional): If True, downloads the dataset from the internet and puts it in the root directory
    """

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.transform = transform
        self.train = train

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please use download=True to download it.")

        if self.train:
            self.data_folder = osp.join(self.root, "train")
        else:
            self.data_folder = osp.join(self.root, "val")

        self.classes, self.class_to_idx = self._find_classes(
            osp.join(self.root, "train")
        )
        self.samples = self._make_dataset(self.data_folder, self.class_to_idx)
        self.targets = [s[1] for s in self.samples]

    def _check_exists(self):
        return osp.exists(osp.join(self.root, "train")) and osp.exists(
            osp.join(self.root, "val")
        )

    def _download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        # Prompt the user to manually download the dataset
        print("Please manually download the Tiny-ImageNet dataset and extract it to the following path:")
        print(f"  {self.root}")
        print("Download link: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print("After extraction, the directory should contain 'train' and 'val' folders.")

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []

        if self.train:
            for target_class in sorted(class_to_idx.keys()):
                class_index = class_to_idx[target_class]
                target_dir = os.path.join(dir, target_class, "images")
                for root, _, fnames in sorted(os.walk(target_dir)):
                    for fname in sorted(fnames):
                        if fname.endswith(".JPEG"):
                            path = os.path.join(root, fname)
                            item = (path, class_index)
                            images.append(item)
        else:
            val_annotations_file = os.path.join(dir, "val_annotations.txt")
            if not os.path.exists(val_annotations_file):
                raise RuntimeError(f"Validation annotation file not found: {val_annotations_file}")

            # Read image labels from the val_annotations.txt file
            with open(val_annotations_file, "r") as f:
                val_annotations = f.readlines()

            # Parse the annotation file to get the mapping from image names to classes
            image_to_class = {}
            for line in val_annotations:
                parts = line.strip().split()
                if len(parts) >= 2: 
                    image_filename = parts[0]
                    class_id = parts[1]
                    if class_id in class_to_idx:
                        image_to_class[image_filename] = class_to_idx[class_id]

            images_dir = os.path.join(dir, "images")
            for filename in sorted(os.listdir(images_dir)):
                if filename in image_to_class and filename.endswith(".JPEG"):
                    path = os.path.join(images_dir, filename)
                    item = (path, image_to_class[filename])
                    images.append(item)

        return images

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

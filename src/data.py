from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

class WeakSegmentationDataset(OxfordIIITPet):
    def __init__(self, root, split, transform=None):
        super().__init__(root, split, transform=transform)

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        return image, mask

    def __len__(self):
        return len(self.samples)


class WeakSegmentationDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return iter(DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle))
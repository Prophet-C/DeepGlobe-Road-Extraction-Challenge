import numpy as np
from sklearn.metrics import confusion_matrix


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Pixel_Precision(self):
        self.precision = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1])
        return self.precision

    def Pixel_Recall(self):
        self.recall = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        return self.recall

    def Pixel_F1(self):
        f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        return f1

    def Intersection_over_Union(self, class_index=1):
        IoU = self.confusion_matrix[class_index, class_index] / (self.confusion_matrix[class_index, class_index] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1] + 1e-10)
        return IoU

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def add_batch_sklearn(self, gt_image, pre_image):
        temp = confusion_matrix(gt_image.reshape(-1), pre_image.reshape(-1), labels=[0, 1])

        self.confusion_matrix += temp




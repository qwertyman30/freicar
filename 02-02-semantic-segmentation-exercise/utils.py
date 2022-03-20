import numpy as np


class ConfusionMatrix:
    def __init__(self, pred, actual):
        self.pred = pred
        self.actual = actual

    def construct(self):
        assert (self.pred.shape == self.actual.shape)
        assert (self.pred[self.pred > 3].size == 0)
        assert (self.pred[self.pred < 0].size == 0)
        assert (self.actual[self.actual > 3].size == 0)
        assert (self.actual[self.actual < 0].size == 0)

        # -------------converting into 1d array and then finding the frequency of each class-------------
        self.pred = self.pred.reshape((245760,))
        # storing the frequency of each class present in the predicted mask
        self.pred_count = np.bincount(self.pred, weights=None, minlength=4)  # A

        self.actual = self.actual.reshape((245760,))
        # storing the frequency of each class present in the actual mask
        self.actual_count = np.bincount(self.actual, weights=None, minlength=4)  # B

        # store the category of every pixel
        temp = self.actual * 4 + self.pred

        # frequency count of temp gives the confusion matrix 'cm' in 1d array format
        self.cm = np.bincount(temp, weights=None, minlength=16)
        # reshaping the confusion matrix from 1d array to (no.of classes X no. of classes)
        self.cm = self.cm.reshape((4, 4))

        # diagonal values of cm correspond to those pixels which belong to same class in both predicted and actual mask
        self.Nr = np.diag(self.cm)  # A ⋂ B
        self.Dr = self.pred_count + self.actual_count - self.Nr  # A ⋃ B
        return self.cm

    def computeMiou(self):
        individual_iou = self.Nr / self.Dr  # (A ⋂ B)/(A ⋃ B)
        miou = np.nanmean(individual_iou)  # nanmean is used to n
        return miou

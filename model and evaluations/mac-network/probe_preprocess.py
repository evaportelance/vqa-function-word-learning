import time
import os
import random
import json
import pickle
import numpy as np
from config import config
from preprocess import SymbolDict, Preprocesser, bold, bcolored, writeline, writelist, vectorize2DList, vectorize3DList


# New ProbePreprocessor class which inherits from the original Preprocessor class from MAC with changes made in order to integrate Probes

class ProbePreprocessor(Preprocesser):
    def __init__(self):
        super().__init__()

    def writeProbePreds(self, res, tier, epoch, batchNum, suffix=""):
        if res is None:
            return
        preds = res["preds"]
        sortedPreds = sorted(preds, key = lambda instance: instance["index"])
        with open(config.answersFile(tier + "-" + str(epoch) + "_" + str(batchNum) + suffix ), "w") as outFile:
            writelist(outFile, ["epoch", "batchNum","index", "question", "answer", "prediction", "image_index"])
            for instance in sortedPreds:
                writelist(outFile, [str(epoch), str(batchNum), instance["index"], instance["question"], instance["answer"], instance["prediction"], instance["imageId"]])

    def readTier(self, tier, train):
        if "probe" in tier:
            probe_name = tier[5:] + "_val_questions.json"
            imagesFilename = config.imagesFile("val")
            datasetFilename = os.path.join(config.dataBasedir, "CLEVR_PROBES", probe_name)
        else:
            imagesFilename = config.imagesFile(tier)
            datasetFilename = config.datasetFile(tier)
        instancesFilename = config.instancesFile(tier)

        instances = self.readData(datasetFilename, instancesFilename, train)

        images = {"imagesFilename": imagesFilename}
        if config.dataset == "NLVR":
            images["imageIdsFilename"] = config.imagesIdsFile(tier)

        return {"instances": instances, "images": images, "train": train}

    '''
    Reads data in datasetFilename, and creates json dictionary.
    If instancesFilename exists, restore dictionary from this file.
    Otherwise, save created dictionary to instancesFilename.
    '''
    def readData(self, datasetFilename, instancesFilename, train):
        # data extraction
        datasetReader = {
            "CLEVR": self.readCLEVR,
            "CLEVRCHILDES": self.readCLEVR,
            "CLEVRnoANDnoEQUAL": self.readCLEVR,
            "CLEVR10AND10EQUAL": self.readCLEVR,
            "CLEVRnoORnoEQUAL": self.readCLEVR,
            "NLVR": self.readNLVR
        }

        return datasetReader[config.dataset](datasetFilename, instancesFilename, train)


    # Adding probeX datasets
    def readDataset(self, suffix = "", hasTrain = True):
        dataset = {"train": None, "evalTrain": None, "val": None, "test": None, "probeAND": None, "probeOR": None, "probeMORE": None, "probeLESS": None, "probeBEHIND": None, "probeFRONT": None, "probeSAME": None, "probeAND2":None, "probeOR2":None}
        if hasTrain:
            dataset["train"] = self.readTier("train" + suffix, train = True)
        dataset["val"] = self.readTier("val" + suffix, train = False)
        dataset["test"] = self.readTier("test" + suffix, train = False)
        dataset["probeAND"] = self.readTier("probeAND" + suffix, train = False)
        dataset["probeOR"] = self.readTier("probeOR" + suffix, train = False)
        dataset["probeMORE"] = self.readTier("probeMORE" + suffix, train = False)
        dataset["probeLESS"] = self.readTier("probeLESS" + suffix, train = False)
        dataset["probeBEHIND"] = self.readTier("probeBEHIND" + suffix, train = False)
        dataset["probeFRONT"] = self.readTier("probeFRONT" + suffix, train = False)
        dataset["probeSAME"] = self.readTier("probeSAME" + suffix, train = False)
        dataset["probeAND2"] = self.readTier("probeAND2" + suffix, train = False)
        dataset["probeOR2"] = self.readTier("probeOR2" + suffix, train = False)

        if hasTrain:
            dataset["evalTrain"] = {}
            for k in dataset["train"]:
                dataset["evalTrain"][k] = dataset["train"][k]
            dataset["evalTrain"]["train"] = False

        return dataset

    def prepareData(self, data, train, filterKey = None, noBucket = False):
        filterDefault = {"maxQLength": 0, "maxPLength": 0, "onlyChain": False, "filterOp": 0}

        filterTrain = {"maxQLength": config.tMaxQ, "maxPLength": config.tMaxP,
                       "onlyChain": config.tOnlyChain, "filterOp": config.tFilterOp}

        filterVal = {"maxQLength": config.vMaxQ, "maxPLength": config.vMaxP,
                     "onlyChain": config.vOnlyChain, "filterOp": config.vFilterOp}

        filters = {"train": filterTrain, "evalTrain": filterTrain,
                   "val": filterVal, "test": filterDefault, "probeAND": filterDefault, "probeOR": filterDefault, "probeMORE": filterDefault, "probeLESS": filterDefault, "probeBEHIND": filterDefault, "probeFRONT": filterDefault, "probeSAME": filterDefault, "probeAND2": filterDefault, "probeOR2": filterDefault}

        if filterKey is None:
            fltr = filterDefault
        else:
            fltr = filters[filterKey]

        # split data when finetuning on validation set
        if config.trainExtra and config.extraVal and (config.finetuneNum > 0):
            if train:
                data = data[:config.finetuneNum]
            else:
                data = data[config.finetuneNum:]

        typeFilter = config.typeFilters[fltr["filterOp"]]
        # filter specific settings
        if fltr["onlyChain"]:
            data = [d for d in data if all((len(inputNum) < 2) for inputNum in d["programInputs"])]
        if fltr["maxQLength"] > 0:
            data = [d for d in data if len(d["questionSeq"]) <= fltr["maxQLength"]]
        if fltr["maxPLength"] > 0:
            data = [d for d in data if len(d["programSeq"]) <= fltr["maxPLength"]]
        if len(typeFilter) > 0:
            data = [d for d in data if d["programSeq"][-1] not in typeFilter]

        # run on subset of the data. If 0 then use all data
        num = config.trainedNum if train else config.testedNum
        if filterKey in {"probeAND", "probeOR", "probeMORE", "probeLESS", "probeBEHIND", "probeFRONT", "probeSAME", "probeAND2", "probeOR2"}:
            num = 0
        # retainVal = True to retain same sample of validation across runs
        if (not train) and (not config.retainVal):
            random.shuffle(data)
        if num > 0:
            data = data[:num]
        # set number to match dataset size
        if train:
            config.trainedNum = len(data)
        else:
            config.testedNum = len(data)

        # bucket
        buckets = self.bucketData(data, noBucket = noBucket)

        # vectorize
        return [self.vectorizeData(bucket) for bucket in buckets]

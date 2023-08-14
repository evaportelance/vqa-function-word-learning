from __future__ import division
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")
import sys
import os
import time
import math
import random
try:
    import Queue as queue
except ImportError:
    import queue
import threading
import h5py
import json
import numpy as np
import tensorflow as tf
from termcolor import colored, cprint

from config import config, loadDatasetConfig, parseArgs
from probe_preprocess_andor import ProbePreprocessor, bold, bcolored, writeline, writelist
from model import MACnet
from collections import defaultdict

## import all functions from main that are unchanged and called below
from main import writePreds, setSession, setSavers, chooseTrainingData, runEvaluation, improveEnough, better, trimData, getLength, getBatches, openImageFiles, closeImageFiles, loadImageBatch, loadImageBatches, alternateData, initStats, updateStats, statsToStr, printTierResults, printDatasetResults, lastLoggedEpoch


# Writes log header to file
def logInit():
    with open(config.logFile(), "a+") as outFile:
        writeline(outFile, config.expName)
        headers = ["epoch", "trainAcc", "valAcc", "trainLoss", "valLoss"]
        if config.evalTrain:
            headers += ["evalTrainAcc", "evalTrainLoss"]
        headers += ["time", "lr"]
        headers += ["AND2probeAcc", "OR2probeAcc"]

        writelist(outFile, headers)
        # lr assumed to be last

def probelogInit():
    filename = "probe-results-" + config.expName + ".csv"
    probelogfile = os.path.join(config.logPath, config.expName, filename)
    with open(probelogfile, "a+") as outFile:
        writeline(outFile, config.expName)
        headers = ["epoch", "batchNum"]
        headers += ["AND2probeAcc", "OR2probeAcc"]

        writelist(outFile, headers)
    return probelogfile

# Writes log record to file
def logRecord(epoch, epochTime, lr, trainRes, evalRes, evalProbeRes):
    with open(config.logFile(), "a+") as outFile:
        record = [epoch, trainRes["acc"], evalRes["val"]["acc"], trainRes["loss"], evalRes["val"]["loss"]]
        if config.evalTrain:
            record += [evalRes["evalTrain"]["acc"], evalRes["evalTrain"]["loss"]]
        record += [epochTime, lr]
        record += [evalProbeRes["probeAND2"]["acc"], evalProbeRes["probeOR2"]["acc"]]

        writelist(outFile, record)


### Defining new functions for probe analysis
def printProbeDatasetResults(trainRes, evalRes, evalProbeRes):
    printTierResults("Training", trainRes, "magenta")
    printTierResults("Training EMA", evalRes["evalTrain"], "red")
    printTierResults("Validation", evalRes["val"], "cyan")
    printTierResults("AND2 Probe", evalProbeRes["probeAND2"], "red")
    printTierResults("OR2 Probe", evalProbeRes["probeOR2"], "red")


def writeProbePreds(preprocessor, evalRes, epoch = 0, batchNum = 0):
    preprocessor.writeProbePreds(evalRes["probeAND2"], "probeAND2", epoch, batchNum)
    preprocessor.writeProbePreds(evalRes["probeOR2"], "probeOR2", epoch, batchNum)

def loadWeights(sess, saver, init):
    if config.restoreEpoch > 0 or config.restore:
        # restore last epoch only if restoreEpoch isn't set
        if config.restoreEpoch == 0:
            # restore last logged epoch
            config.restoreEpoch, config.lr = lastLoggedEpoch()
        print(bcolored("Restoring epoch {} and lr {}".format(config.restoreEpoch, config.lr),"cyan"))
        print(bcolored("Restoring weights", "blue"))
        saver.restore(sess, config.weightsFile(config.restoreEpoch))
        epoch = config.restoreEpoch
    else:
        print(bcolored("Initializing weights", "blue"))
        sess.run(init)
        logInit()
        epoch = 0

    return epoch


def runProbeEvaluation(sess, model, data, epoch, probelogfile, batchNum = 0, getAtt = None):
    if getAtt is None:
        getAtt = config.getAtt
    res = {"probeAND2": None, "probeOR2": None }

    if data is not None:
        res["probeAND2"] = runProbeEpoch(sess, model, data["probeAND2"], train = False, epoch = epoch, getAtt = getAtt)
        res["probeOR2"] = runProbeEpoch(sess, model, data["probeOR2"], train = False, epoch = epoch, getAtt = getAtt)

        with open(probelogfile, "a+") as outFile:
            record = [epoch, batchNum]
            record += [res["probeAND2"]["acc"], res["probeOR2"]["acc"]]
            writelist(outFile, record)
    return res


imagesQueue = queue.Queue(maxsize = 20) # config.tasksNum
inQueue = queue.Queue(maxsize = 1)
outQueue = queue.Queue(maxsize = 1)

# Runs a worker thread(s) to load images while training .
class StoppableThread(threading.Thread):
    # Thread class with a stop() method. The thread itself has to check
    # regularly for the stopped() condition.

    def __init__(self, images, batches): # i
        super(StoppableThread, self).__init__()
        # self.i = i
        self.images = images
        self.batches = batches
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while not self.stopped():
            try:
                batchNum = inQueue.get(timeout = 60)
                nextItem = loadImageBatches(self.images, self.batches, batchNum, int(config.taskSize / 2))
                outQueue.put(nextItem)
                # inQueue.task_done()
            except:
                pass
        # print("worker %d done", self.i)

def loaderRun(images, batches):
    batchNum = 0
    while batchNum < len(batches):
        nextItem = loadImageBatches(images, batches, batchNum, config.taskSize)
        assert len(nextItem) == min(config.taskSize, len(batches) - batchNum)
        batchNum += config.taskSize
        imagesQueue.put(nextItem)


'''
Runs an epoch with model and session over the data.
1. Batches the data and optionally mix it with the extra alterData.
2. Start worker threads to load images in parallel to training.
3. Runs model for each batch, and gets results (e.g. loss,  accuracy).
4. Updates and prints statistics based on batch results.
5. Once in a while (every config.saveEvery), save weights.

Args:
    sess: TF session to run with.

    model: model to process data. Has runBatch method that process a given batch.
    (See model.py for further details).

    data: data to use for training/evaluation.

    epoch: epoch number.

    saver: TF saver to save weights

    calle: a method to call every number of iterations (config.calleEvery)

    alterData: extra data to mix with main data while training.

    getAtt: True to return model attentions.
'''
def runEpoch(sess, model, data, train, epoch, probelogfile = None, preprocessor = None, saver = None, calle = None, alterData = None, getAtt = False):
    # initialization
    startTime0 = time.time()
    train_data = data["main"]["train"] #better than outside argument

    stats = initStats()
    preds = []

    # open image files
    openImageFiles(train_data["images"])

    ## prepare batches
    buckets = train_data["data"]
    dataLen = sum(getLength(bucket) for bucket in buckets)

    # make batches and randomize
    batches = []
    for bucket in buckets:
        batches += getBatches(bucket, batchSize = config.batchSize)
    random.shuffle(batches)

    # alternate with extra data
    if train and alterData is not None:
        batches, dataLen = alternateData(batches, alterData, dataLen)

    # start image loaders
    if config.parallel:
        loader = threading.Thread(target = loaderRun, args = (train_data["images"], batches))
        loader.daemon = True
        loader.start()

    for batchNum, batch in enumerate(batches):
        startTime = time.time()

        # prepare batch
        batch = trimData(batch)

        # load images batch
        if config.parallel:
            if batchNum % config.taskSize == 0:
                imagesBatches = imagesQueue.get()
            imagesBatch = imagesBatches[batchNum % config.taskSize] # len(imagesBatches)
        else:
            imagesBatch = loadImageBatch(train_data["images"], batch)
        for i, imageId in enumerate(batch["imageIds"]):
            assert imageId == imagesBatch["imageIds"][i]


        # run batch
        res = model.runBatch(sess, batch, imagesBatch, train, getAtt)

        # update stats
        stats = updateStats(stats, res, batch)
        preds += res["preds"]

        # save weights
        if saver is not None:
            if batchNum > 0 and batchNum % config.saveEvery == 0:
                print("")
                print(bold("saving weights"))
                saver.save(sess, config.weightsFile(epoch))

                    # run probe eval every so many batches when training
        # if train:
        #     if epoch < 3 and (batchNum+1) % 2000 == 0:
        #         print(bold("Running probes evaluations..."))
        #         evalProbeRes = runProbeEvaluation(sess, model, data["main"], epoch = epoch, probelogfile = probelogfile, batchNum= batchNum)
        #         print(bold("Writing probes predictions..."))
        #         writeProbePreds(preprocessor, evalProbeRes, epoch, batchNum)

        # calle
        if calle is not None:
            if batchNum > 0 and batchNum % config.calleEvery == 0:
                calle()

    closeImageFiles(train_data["images"])

    if config.parallel:
        loader.join() # should work

    return {"loss": stats["loss"],
            "acc": stats["acc"],
            "preds": preds
            }

def runProbeEpoch(sess, model, data, train, epoch, getAtt = False):
    # train = data["train"] better than outside argument

    # initialization
    startTime0 = time.time()

    stats = initStats()
    preds = []

    # open image files
    openImageFiles(data["images"])

    ## prepare batches
    buckets = data["data"]
    dataLen = sum(getLength(bucket) for bucket in buckets)

    # make batches and randomize
    batches = []
    for bucket in buckets:
        batches += getBatches(bucket, batchSize = config.batchSize)
    random.shuffle(batches)

    for batchNum, batch in enumerate(batches):
        startTime = time.time()

        # prepare batch
        batch = trimData(batch)

        # load images batch
        imagesBatch = loadImageBatch(data["images"], batch)
        for i, imageId in enumerate(batch["imageIds"]):
            assert imageId == imagesBatch["imageIds"][i]

        # run batch
        res = model.runBatch(sess, batch, imagesBatch, train, getAtt)

        # update stats
        stats = updateStats(stats, res, batch)
        preds += res["preds"]

    closeImageFiles(data["images"])

    return {"loss": stats["loss"],
            "acc": stats["acc"],
            "preds": preds
            }


def set_random_seed():
    SEED = int(config.expName[-1])
    print("SEED", SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


'''
Trains/evaluates the model:
1. Set GPU configurations.
2. Preprocess data: reads from datasets, and convert into numpy arrays.
3. Builds the TF computational graph for the MAC model.
4. Starts a session and initialize / restores weights.
5. If config.train is True, trains the model for number of epochs:
    a. Trains the model on training data
    b. Evaluates the model on training / validation data, optionally with
       exponential-moving-average weights.
    c. Prints and logs statistics, and optionally saves model predictions.
    d. Optionally reduces learning rate if losses / accuracies don't improve,
       and applies early stopping.
6. If config.test is True, runs a final evaluation on the dataset and print
   final results!
'''
def main():
    with open(config.configFile(), "a+") as outFile:
        json.dump(vars(config), outFile)

    # set random seeds
    set_random_seed()

    # set gpus
    if config.gpus != "":
        config.gpusNum = len(config.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    #tf.logging.set_verbosity(tf.logging.ERROR)
    # EP - made change for compatipility with tf v2
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # process data
    print(bold("Preprocess data..."))
    start = time.time()
    preprocessor = ProbePreprocessor()
    data, embeddings, answerDict = preprocessor.preprocessData()
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    # build model
    # EP adding for tf v2 compatibility
    tf.compat.v1.disable_eager_execution()
    print(bold("Building model..."))
    start = time.time()
    model = MACnet(embeddings, answerDict)
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    # initializer
    init = tf.compat.v1.global_variables_initializer()

    # savers
    savers = setSavers(model)
    saver, emaSaver = savers["saver"], savers["emaSaver"]

    # sessionConfig
    sessionConfig = setSession()

    with tf.compat.v1.Session(config = sessionConfig) as sess:

        # ensure no more ops are added after model is built
        sess.graph.finalize()

        # restore / initialize weights, initialize epoch variable
        epoch = loadWeights(sess, saver, init)

        if config.train:
            start0 = time.time()

            bestEpoch = epoch
            bestRes = None
            prevRes = None

            # Get probe performance at before training
            probelogfile = probelogInit()
            print(bold("Running probes evaluations..."))
            evalProbeRes = runProbeEvaluation(sess, model, data["main"], epoch, probelogfile)
            print(bold("Writing probes predictions..."))
            writeProbePreds(preprocessor, evalProbeRes)

            # epoch in [restored + 1, epochs]
            for epoch in range(config.restoreEpoch + 1, config.epochs + 1):
                print(bcolored("Training epoch {}...".format(epoch), "green"))
                start = time.time()

                # train
                trainingData, alterData = chooseTrainingData(data)
                trainRes = runEpoch(sess, model, data, train = True, epoch = epoch, probelogfile = probelogfile, preprocessor = preprocessor, saver = saver, alterData = alterData)

                # save weights
                saver.save(sess, config.weightsFile(epoch))
                if config.saveSubset:
                    subsetSaver.save(sess, config.subsetWeightsFile(epoch))

                # load EMA weights
                if config.useEMA:
                    print(bold("Restoring EMA weights"))
                    emaSaver.restore(sess, config.weightsFile(epoch))

                # evaluation
                evalRes = runEvaluation(sess, model, data["main"], epoch)
                extraEvalRes = runEvaluation(sess, model, data["extra"], epoch,
                    evalTrain = not config.extraVal)

                # Get probe performance
                print(bold("Running probes evaluations..."))
                evalProbeRes = runProbeEvaluation(sess, model, data["main"], epoch, probelogfile)
                print(bold("Writing probes predictions..."))
                writeProbePreds(preprocessor, evalProbeRes, epoch)

                # restore standard weights
                if config.useEMA:
                    print(bold("Restoring standard weights"))
                    saver.restore(sess, config.weightsFile(epoch))

                print("")

                epochTime = time.time() - start
                print("took {:.2f} seconds".format(epochTime))

                # print results
                printProbeDatasetResults(trainRes, evalRes, evalProbeRes)

                # stores predictions and optionally attention maps
                if config.getPreds:
                    print(bcolored("Writing predictions...", "white"))
                    writePreds(preprocessor, evalRes, extraEvalRes)

                logRecord(epoch, epochTime, config.lr, trainRes, evalRes, evalProbeRes)

                # update best result
                # compute curr and prior
                currRes = {"train": trainRes, "val": evalRes["val"]}
                curr = {"res": currRes, "epoch": epoch}

                if bestRes is None or better(currRes, bestRes):
                    bestRes = currRes
                    bestEpoch = epoch

                prior = {"best": {"res": bestRes, "epoch": bestEpoch},
                         "prev": {"res": prevRes, "epoch": epoch - 1}}

                # lr reducing
                if config.lrReduce:
                    if not improveEnough(curr, prior, config.lr):
                        config.lr *= config.lrDecayRate
                        print(colored("Reducing LR to {}".format(config.lr), "red"))

                # early stopping
                if config.earlyStopping > 0:
                    if epoch - bestEpoch > config.earlyStopping:
                        break

                # update previous result
                prevRes = currRes

            # reduce epoch back to the last one we trained on
            epoch -= 1
            print("Training took {:.2f} seconds ({:} epochs)".format(time.time() - start0,
                epoch - config.restoreEpoch))

        if config.finalTest:
            print("Testing on epoch {}...".format(epoch))

            start = time.time()
            if epoch > 0:
                if config.useEMA:
                    emaSaver.restore(sess, config.weightsFile(epoch))
                else:
                    saver.restore(sess, config.weightsFile(epoch))

            evalRes = runEvaluation(sess, model, data["main"], epoch, evalTest = True)
            extraEvalRes = runEvaluation(sess, model, data["extra"], epoch,
                evalTrain = not config.extraVal, evalTest = True)

            print("took {:.2f} seconds".format(time.time() - start))
            printDatasetResults(None, evalRes, extraEvalRes)

            print("Writing predictions...")
            writePreds(preprocessor, evalRes, extraEvalRes)

        print(bcolored("Done!","white"))

if __name__ == '__main__':
    parseArgs()
    loadDatasetConfig[config.dataset]()
    main()

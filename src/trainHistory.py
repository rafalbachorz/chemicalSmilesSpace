import tensorflow.keras as keras
import tensorflow.keras.backend as K
import math

class trainHistory(keras.callbacks.Callback):
    def __init__(self, initialLR, drop, epochsDrop, dtime):
        super().__init__()
        self.__initialLR=initialLR
        self.__drop=drop
        self.__epochsDrop=epochsDrop
        self.__dtime=dtime
        self.__losses=[]
        self.__accuracy=[]
        self.__file=None

    def on_train_begin(self, logs={}):
        fileName=self.__dtime+"_diagnostics.txt"
        self.__file=open(fileName, 'a')
        K.set_value(self.model.optimizer.lr, self.__initialLR)
        return

    def on_train_end(self, logs={}):
        self.__file.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        # lr = optimizer.lr
        # print('\nLR epoch beg: {:.6f}\n'.format(lr))
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        print('\nLR before the epoch: {:.6f}\n'.format(lr))
        return

    def on_epoch_end(self, epoch, logs={}):
        loss=logs.get("loss")
        self.__losses.append(loss)
        accuracy=logs.get("acc")
        self.__accuracy.append(accuracy)
        optimizer = self.model.optimizer
        # self.model.optimizer.lr = self._step_decay(epoch)
        lr=K.eval(optimizer.lr)

        print('\nEpoch: {:.6f}; LR at the end of epoch: {:.6f}\n'.format(epoch, lr))
        lr=self.__step_decay(epoch)
        print('\nSetting up new LR: {:.6f}\n'.format(lr))
        K.set_value(self.model.optimizer.lr, lr)
        if ((epoch+1) % self.__epochsDrop==0):
            print('\nStoring model/weights on file...\n')
            self.__saveModel(epoch, loss)
        self.__storeDiagnostics(epoch, logs)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def __step_decay(self, epoch):
        lrate = self.__initialLR * math.pow(self.__drop, math.floor((1 + epoch) / self.__epochsDrop))
        return lrate

    def __saveModel(self, epoch, loss):
        model_json = self.model.to_json()
        modelName=self.__dtime+"_"+str(epoch)+"_"+"{:6.4f}".format(loss)
        with open(modelName + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(modelName + '.h5')

    def __storeDiagnostics(self, epoch, logs):
        loss=logs.get("loss")
        accuracy = logs.get("acc")
        self.__file.writelines("{};{:6.4f};{:6.4f}".format(epoch, loss, accuracy)+"\n")
        self.__file.flush()

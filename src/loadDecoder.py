import trainSmiles
import predictiveModel
from keras.models import Model

def loadCNNdecoder(nCharInSmiles, nCharSet, nStatic, encoderDimensions, decoderDimensions, lr, modelFile):
    #prepareEncoderCNNDynamic(nCharInSmiles, nCharSet, k, lr, variational, showArch)
    encoder = trainSmiles.prepareEncoderCNNDynamic(nCharInSmiles, nCharSet, encoderDimensions, lr, True, True)[0]
    #prepareDecoderCNNDynamic(nCharInSmiles, nCharSet, k, lr, showArch)
    decoder = trainSmiles.prepareDecoderCNNDynamic(nCharInSmiles, nCharSet, decoderDimensions, lr, True)
    encoderOutput = encoder.get_layer('encoderOutput').output
    decoderOutput = decoder(encoderOutput)
    autoencoder = Model(inputs=encoder.input, outputs=decoderOutput)

    autoencoder.load_weights(modelFile)
    decoder.load_weights(modelFile, by_name=True)
    encoder.load_weights(modelFile, by_name=True)

    return encoder, decoder, autoencoder
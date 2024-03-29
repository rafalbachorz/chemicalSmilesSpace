import pickle
def picklePredictiveModel(modelWeightsFile, nCharInSmiles, nCharSet, nStatic, nLatent, scaler, char2indices, 
                            indices2char, fullModelFileName, smilesCodes, hyperparameters):
        predictiveModel = {}
        predictiveModel['model'] = modelWeightsFile
        predictiveModel['nCharSmiles'] = nCharInSmiles
        predictiveModel['nCharSet'] = nCharSet
        predictiveModel['nStatic'] = nStatic
        predictiveModel['scaler'] = scaler
        predictiveModel['nLatent'] = nLatent
        predictiveModel['char2indices'] = char2indices
        predictiveModel['indices2char'] = indices2char
        predictiveModel['smilesCodes'] = smilesCodes
        predictiveModel['hyperparameters'] = hyperparameters

        with open(fullModelFileName, 'wb') as f:
            pickle.dump(predictiveModel, file=f)

def unpicklePredictiveModel(fullModelFileName):
        with open(fullModelFileName, 'rb') as f:
            predictiveModel = pickle.load(file=f)
        return predictiveModel
        

class predictiveModel:
    def __init__(self, model, history, nCharInSmiles, nCharSet, nStatic, scaler, char2indices, indices2char):
        self.model = model
        self.history = history
        self.nCharSmiles = nCharInSmiles
        self.nCharSet = nCharSet
        self.nStatic = nStatic
        self.scaler = scaler
        self.char2indices = char2indices
        self.indices2char = indices2char
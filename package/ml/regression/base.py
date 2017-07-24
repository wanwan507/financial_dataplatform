from ml.base import BaseModel


class Regression(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._features = None
        self._target = None

    # train the model with given data set
    def train(self, data):
        self._features = data["train"]
        self._target = data["target"]
        self._model.fit(self._features, self._target)

    # train the model with given data set
    def getParameterDef(self):
        pass

    def setParameter(self, parameter):
        pass

    # predict the model with given dataset
    def predict(self, data):
        return self._model.predict(data)

    def predictViz(self, scale):
        # Predict Viz only available for one dimensional dataset
        if len(self._features[0]) != 1:
            return None

        result = dict()
        result["predict"] = list()
        result["data"] = list()

        for i in range(0, len(self._features)):
            item = dict()
            item["x"] = self._features[i][0]
            item["y"] = self._target[i]
            result["data"].append(item)

        aarange = dict()
        aarange["xmin"] = self._features[0][0]
        aarange["xmax"] = self._features[0][0]

        for item in self._features:
            if item[0] > aarange["xmax"]:
                aarange["xmax"] = item[0]
            if item[0] < aarange["xmin"]:
                aarange["xmin"] = item[0]

        xstep = (float(aarange["xmax"]) - float(aarange["xmin"])) / scale

        for x in range(0, scale):
            dx = aarange["xmin"] + x * xstep

            onePredict = self.predict([[dx]])
            record = dict()
            record["x"] = dx
            record["y"] = onePredict[0]
            result["predict"].append(record)

        return result

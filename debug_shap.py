import pickle
model = pickle.load(open("models/clinical_rf.pkl", "rb"))
print(model.classes_)

# Load the pickled label encoders
with open('label_encoders.pkl', 'rb') as file:
    loaded_label_encoders = pickle.load(file)
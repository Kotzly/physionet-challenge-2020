import numpy as np
from physionet_challenge.processing.features import baseline_features

def run_classifier_for_subject(data, header_data, artifacts):
    # Use your classifier here to obtain a label and score for each class.
    model = artifacts['model']
    imputer = artifacts['imputer']
    classes = artifacts['classes']
    scaler = artifacts['scaler']

    features = np.asarray(baseline_features(data, header_data))
    feats_reshape = features.reshape(1, -1)
    feats_reshape = imputer.transform(feats_reshape)
    feats_reshape = scaler.transform(feats_reshape)
    current_score = model.predict(feats_reshape)[0]
    current_label = (current_score > .5).astype(int)
    
    return current_label, current_score, classes

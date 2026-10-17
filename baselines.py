from nam_code.helper.nam_train_test_save_load import load_full_model,load_data
import torch 
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

def compute_shap_lime(model_pth, dataset, batch_size, device, sample_indices):
    train_loader, test_loader, input_size = load_data(dataset, batch_size=batch_size)
    model = load_full_model(model_pth, input_size=input_size, device="cpu")

    X_train = np.array(torch.cat([x for x, _ in train_loader]))
    background = X_train[np.random.choice(len(X_train), 100, replace=False)]

    # SHAP
    shapley_vals = np.empty((len(sample_indices), len(X_train[0])))

    for indx, sample_index in enumerate(sample_indices):
        X_batch, y_batch= list(test_loader)[sample_index]
        x = X_batch[0]

        with torch.no_grad():
            out = model(x[None]).numpy().reshape(-1)[0]

        sha = np.empty(len(x))

        for i in range(len(x)):
            plc = np.tile(x, (len(background), 1))
            plc[:, i] =  background[:,i]
            with torch.no_grad():
                bk = model(torch.as_tensor(plc)).numpy().reshape(-1).mean()
            sha[i] = out - bk
            
        for indy, val in enumerate(sha):
            shapley_vals[indx, indy] = val

    # LIME
    X = np.random.normal(size=(len(sample_indices), len(X_train[0])))

    def predict(X):
        with torch.no_grad():
            out = model(torch.as_tensor(X, dtype=torch.float32))

        return out.numpy().reshape(-1)

    explainer = LimeTabularExplainer(X_train, mode="regression", discretize_continuous=False, sample_around_instance=True)
    
    lime_vals = np.zeros((len(X), X_train.shape[1]))

    for r, x in enumerate(X):
        exp = explainer.explain_instance(x, predict, num_features=X_train.shape[1],
                                         num_samples=5000)
        
        for idx, coef in exp.as_map()[1]: 
            lime_vals[r, idx] = coef


    # Normalise
    plc_out = np.abs(shapley_vals)
    max_val =  plc_out.max(axis=1, keepdims=True)
    max_val = np.where(max_val==0, 1.0, max_val)

    lime_out = np.abs(lime_vals)
    max_val_lime = lime_out.max(axis=1, keepdims=True)
    max_val_lime = np.where(max_val_lime==0, 1.0, max_val_lime)

    return plc_out / max_val, lime_out / max_val_lime


if __name__ == "__main__":
    dataset = "breast_cancer"
    batch_size = 1
    device = "cpu"
    sample_indices = range(30)
    model_pth = "C:/Users/Spark/Desktop/tu_delft/formal_methods/nam_code/models/breast_cancer/nam_full.pth"

    test = compute_shap_lime(model_pth,
                        dataset,
                        batch_size,
                        device,
                        sample_indices,
                        )

    plt.imshow(test)
    plt.show()

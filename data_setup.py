##### OK create function, split data, do trainsformations, get into dataset, dataloders, return class names and the 0loaders
import torch
import kagglehub
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd


##        Index(['id', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
##       'ConvexArea', 'EquivDiameter', 'Extent', 'Perimeter', 'Roundness',
##       'AspectRation', 'Class'],
##      dtype='object')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float).to(DEVICE)
        self.Y = torch.tensor(Y, dtype=torch.float).to(DEVICE)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#BATCH_SIZE, NUM_WORKERS, DEVICE
# remeber dataset is tabular no image


def setup_data(BATCH_SIZE, NUM_WORKERS, DEVICE):
    print('Setting up data...')

    data_set_path = kagglehub.dataset_download("mssmartypants/rice-type-classification")
    cvs_path = f"{data_set_path}/riceClassification.csv"
    print('Dataset downloaded.')
    og_df = pd.read_csv(cvs_path)
    og_df.dropna()
    og_df.drop(['id'], axis=1, inplace=True)



    unique_classes = sorted(og_df['Class'].unique())
    CLASS_NAMES = unique_classes
    features_df = og_df.drop(columns=['Class'], errors='ignore')

    # normalize values
    for column in features_df:
        features_df[column] = features_df[column]/features_df[column].abs().max()

    X = features_df.values
    Y = og_df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) #split fo training

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42) #takes all test values and split in half for val

    print(f"Training set is {X_train.shape[0]} rows")
    print(f"Val set is {X_val.shape[0]} rows")
    print(f"Test set is {X_test.shape[0]} rows")

    ## NOW ACTUALLY MAKING THE DATASET

    training_dataset = dataset(X_train, y_train)
    validation_dataset = dataset(X_val, y_val)
    test_dataset = dataset(X_test, y_test)

    #working with tabular data is soo odd lol


    # DATALOADERS

    train_dl = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    MAX_VALS = og_df.drop(columns=['id', 'Class']).abs().max().to_dict()

    num_features = X_train.shape[1]
    return CLASS_NAMES, train_dl, val_dl, test_dl, num_features, MAX_VALS



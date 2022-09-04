import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F

from adjustText import adjust_text

# device configs
device = torch.device("cpu")

print('Running on', device)

torch.manual_seed(0)
np.random.seed(0)

class GradePredictor(nn.Module):
    def __init__(self, n_students, n_disciplines, n_features=128):
        super().__init__()

        self.year_features = 4
        self.date_features = 4

        # embeddings for students and features
        self.student_embedding = nn.Embedding(n_students, n_features)
        self.discipline_embedding = nn.Embedding(n_disciplines, n_features)
        self.year_embedding = nn.Embedding(100, self.year_features)
        self.date_embedding = nn.Embedding(100, self.date_features)
        

        self.hidden1 = nn.Linear(2*n_features + self.year_features + self.date_features, n_features * 2)
        self.hidden2 = nn.Linear(n_features * 2, n_features)
        self.hidden3 = nn.Linear(n_features, n_features // 2)
        self.hidden4 = nn.Linear(n_features // 2, 1)


        self.hidden1_norm = nn.LayerNorm(n_features * 2)
        self.hidden2_norm = nn.LayerNorm(n_features)
        self.hidden3_norm = nn.LayerNorm(n_features // 2)

        self.do05 = nn.Dropout(0.5)



    def get_discipline_weights(self):
        return self.discipline_embedding.weight.cpu().detach().numpy()
    

    def forward(self, x):

        # unpack input
        student, discipline, year, date = x[:, 0], x[:, 1], x[:, 2], x[:, 3] 
        
        # convert UIDs to embeddings
        student_features, discipline_features = self.student_embedding(student), self.discipline_embedding(discipline)
        year_features, date_features = self.year_embedding(year), self.date_embedding(date)

        # apply dropouts
        student_features = self.do05(student_features)
        discipline_features = self.do05(discipline_features)
        year_features = self.do05(year_features)
        date_features = self.do05(date_features)


        # concat on embedding dim
        x = torch.cat([student_features, discipline_features, year_features, date_features], dim=1)
        
        # first layer
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden1_norm(x)
        
        # second layer
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden2_norm(x)
        
        # third layer
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden3_norm(x)
        
        # output layer
        x = self.hidden4(x)

        # clamp output to the 0-10 interval
        x = torch.sigmoid(x) * 10
        x = x[0]
        

        return x


tsne = TSNE(n_components=2, perplexity=20.0, n_iter=10000)
pca = PCA(n_components=50)


csv = 'csvs/export_note_acs.csv'
dataset = pd.read_csv(csv, encoding='latin-1')

# reduces dataset by removing data until the year 2000
old_data = dataset[(dataset['An Universitar'] < 2000)].index
dataset.drop(old_data, inplace=True)


# removes rows with missing disciplines
dataset.dropna(subset=['Disciplina Scurt'], inplace=True)

# reduces dataset to known disciplines for qualitative evaluation
known_disciplines = ['IntI', 'InsI', 'LS2', 'USO', 'PL', 'MN', 'BE', 'SD', 'M3', 'LS1', 'F', 'PC', 'M2', 'M1', 'EFS1', 'EFS2', 'L', \
    'FCT', 'LS4', 'ADIV', 'ED', 'CN1', 'PCom', 'PP', 'PA', 'LS3', 'TS', 'IOCLA', 'EEA', 'POO', 'AA', \
    'EG', 'IP', 'P', 'BD', 'SO', 'ASC', 'PM', 'EGC', 'CN2', 'RL', 'LFA', 'APD', 'MkI', \
    'TSSC', 'IO', 'SM', 'SMu', 'PPL', 'EIM', 'APPa', 'PRe', 'MP', 'VI', \
    'SPRC', 'Co', 'SOI', 'IA', 'BDI', 'IAu', 'EP', 'SPG']


# replaces 'A's with '10' and strings with NANs in the Nota (grade) column
dataset['Nota'].replace('A', '10', inplace=True)
dataset['Nota'].replace(r'^[A-Za-z\-]*$', np.nan, regex=True, inplace=True)

# drops rows with NANs as their grade
dataset.dropna(subset=['Nota'], inplace=True)

# converts the grade to a float
dataset['Nota'] = dataset['Nota'].astype(float)

# create UIDs for students, disciplines, years of study (1st, second, etc.) and absolute year (date)
dataset['uid_student'] = dataset.groupby(['anonymid']).ngroup()
dataset['uid_discipline'] = dataset.groupby(['Disciplina Scurt']).ngroup()
dataset['uid_year'] = dataset.groupby(['Ciclu de studii', 'An Studii']).ngroup()
dataset['uid_date'] = dataset.groupby(['An Universitar']).ngroup()

n_students = dataset['uid_student'].nunique()
n_disciplines = dataset['uid_discipline'].nunique()

print('Number of students:', n_students)
print('Number of disciplines:', n_disciplines)


# defining the inputs = (student_id, discipline_id, year_id, date_id) and output=(grade)
dataset_targets = torch.Tensor(dataset['Nota'].values.astype(np.long))
dataset_inputs = torch.LongTensor(dataset[['uid_student', 'uid_discipline', 'uid_year', 'uid_date']].values.astype(np.long))
dataset_tensor = torch.utils.data.TensorDataset(dataset_inputs, dataset_targets)

# split into training and validation sets (70/30)
training_set, validation_set = torch.utils.data.random_split(dataset_tensor, [int(len(dataset_tensor) * 0.7), len(dataset_tensor) - int(len(dataset_tensor) * 0.7)])

print(f'Training on {str(len(training_set))} samples')
print(f'Validating on {str(len(validation_set))} samples')

# creating data loaders with BS = 1
training_loader = torch.utils.data.DataLoader(training_set, batch_size = 1, shuffle=True, drop_last=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 1, shuffle=True, drop_last=True)


ACC_STEPS = 64

def train():

    model.train()

    total_loss = 0
    
    optimizer.zero_grad()

    for batch_id, (x_train, y_train) in enumerate(training_loader):

        x_train = x_train.to(device)
        y_train = y_train.to(device)

        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)
        total_loss += loss.item()

        # backwards loss using gradient accumulation
        loss /= ACC_STEPS
        loss.backward()
        
        # optimize weights after ACC_STEPS steps
        if (batch_id + 1) % ACC_STEPS == 0 or batch_id + 1 == len(training_loader):
            optimizer.step()
            optimizer.zero_grad()

        if batch_id % 1000 == 0:
            print(f'Training @{batch_id}')

    return total_loss / len(training_loader)


min_val_loss = 999999

def test():
    global min_val_loss

    model.eval()

    with torch.no_grad():
        total_loss = 0
        err = []

        for batch_id, (x_test, y_test) in enumerate(validation_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_pred = model(x_test)

            loss = criterion(y_pred, y_test)
            total_loss += loss.item()
            
            y_pred = y_pred.cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()

            if batch_id % 1000 == 0:
                print(f'Validating @{batch_id}')

            for i in range(len(y_pred)):
                err.append(abs(y_pred[i] - y_test[i]))
            
        
        if total_loss / len(validation_loader) < min_val_loss:
            min_val_loss = total_loss / len(validation_loader)

            plt.clf()
            plt.figure(figsize=(14,7))
            plt.style.use('seaborn-whitegrid')
            plt.xlabel("Error")
            plt.ylabel("Frequency")

            plt.hist(err, bins=np.arange(0.0, 11.5, 0.5), range=[0.0, 11.0], facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
            plt.xticks(np.arange(0.0, 11.5, 0.5), fontsize=7)
            plt.yticks(range(0, 100000, 5000), fontsize=7)
            plt.savefig('error-distribution.png')
            plt.close()


            torch.save(model.state_dict(), 'checkpoints/' + str(min_val_loss) + '.dat')


            disciplines_embeddings = model.get_discipline_weights()
            pca_result = pca.fit_transform(disciplines_embeddings)
            tsne_result = tsne.fit_transform(pca_result)
            
            tsne_points = {}
            tsne_points['x'] = tsne_result[:, 0]
            tsne_points['y'] = tsne_result[:, 1]

            tsne_points['label'] = []
            tsne_points['full_label'] = []
            for i in range(n_disciplines):
                #print(dataset[(dataset['uid_discipline'] == i)]['Disciplina Scurt'].iloc[0])
                discipline_name = dataset[(dataset['uid_discipline'] == i)]['Disciplina Scurt'].iloc[0]
                full_discipline_name = dataset[(dataset['uid_discipline'] == i)]['Disciplina'].iloc[0]
                tsne_points['label'].append(discipline_name)
                tsne_points['full_label'].append(discipline_name + ':' + full_discipline_name)

                #p1.text(tsne_points['x'][i], tsne_points['y'][i], discipline_name)

            tsne_points['label'] = np.array(tsne_points['label'])
            tsne_points['full_label'] = np.array(tsne_points['full_label'])


            plt.clf()
            plt.figure(figsize=(12, 12))
            
            # filter disciplines
            texts = []
            for i in range(n_disciplines):
                if tsne_points['label'][i] in known_disciplines:
                    plt.scatter(tsne_points['x'][i], tsne_points['y'][i], label=tsne_points['full_label'][i])
                    texts.append(plt.text(tsne_points['x'][i], tsne_points['y'][i], tsne_points['label'][i]))

            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
            

            legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

            plt.savefig('tsne.png', bbox_extra_artists=(legend,), bbox_inches='tight')
            plt.close()


        return total_loss / len(validation_loader)


# Training
model = GradePredictor(n_students, n_disciplines).to(device)
model.load_state_dict(torch.load('checkpoints/2.789827297579973.dat'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
criterion = nn.MSELoss()


for epoch in range(1000):

    train_loss = train()
    print(f'Epoch {epoch}; training loss: {train_loss}')

    test_loss = test()
    print(f'Epoch {epoch}; validation loss: {test_loss}')

    if (epoch+1) % 3 == 0:
        ACC_STEPS *= 2

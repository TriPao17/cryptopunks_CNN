import ast
import concurrent.futures
import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.notebook as tq
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

################ Control Panel ################
build_structure = False
get_punk_data = False
clean_punk_data = False
get_punk_images = False
build_data, train_fraction = False, 0.8
train_network, epo = False, 12
test_data = False
evaluate_performance = False
plot_results = False
create_archive = False

################ Checking for Cuda ################
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on GPU...')
else:
    device = torch.device("cpu")
    print('Running on CPU...')

################ Build Structure ################
if build_structure is True:
    print('Building Structure...')
    if not os.path.exists('punk_data'):
        os.makedirs('punk_data')
    if not os.path.exists('punk_data/training_data'):
        os.makedirs('punk_data/training_data')
    if not os.path.exists('punk_data/networks'):
        os.makedirs('punk_data/networks')
    if not os.path.exists('punk_data/networks/acc_loss_data'):
        os.makedirs('punk_data/networks/acc_loss_data')
    if not os.path.exists('punk_data/networks/graphs'):
        os.makedirs('punk_data/networks/graphs')
    if not os.path.exists('punk_images'):
        os.makedirs('punk_images')

################ Get Punk Data Raw ################
if get_punk_data is True:
    # Initiating parameters to retrieve data
    N = 500
    to_recover = np.arange(0, N).tolist()
    not_recovered = []

    # Function which retrieves punks
    def get_punks(rn):
        # OpenSea API url
        url = 'https://api.opensea.io/api/v1/assets'

        # Query Parameters
        querystring = {'order_direction': 'asc',
                       'offset': '0',
                       'limit': '20',
                       'token_ids': np.arange(rn * 20, rn * 20 + 20),
                       'collection': 'cryptopunks'}

        # Submitting query & Retrieving response
        response_punks = requests.request('GET', url, params=querystring)
        response_dict = json.loads(response_punks.text)
        assets = response_dict.get('assets')

        # Check if the request is fulfilled & appending failed requests
        if assets is None:
            not_recovered.append(rn)

        # Returning retrieved assets
        return assets

    # Status Update
    print('Initialising download...')

    # Setting up Multi-threading loop
    all_assets = []
    while to_recover:
        # Launching threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(get_punks, to_recover)

        # Filtering out empty requests
        for result in results:
            if result is None:
                pass
            else:
                all_assets += result

        # Printing out status
        print(f'{100 - round(len(not_recovered) / N, 2) * 100}%')

        # Setting new items to retrieve
        to_recover = not_recovered
        not_recovered = []

    # Saving with Date
    df_raw = pd.DataFrame(all_assets)
    df_raw.to_csv(f'punk_data/punk_data_raw.csv')

    # Status Update
    print(f'Download Complete. Document saved as ==> punk_data/punk_data_raw.csv')

################ Clean Punk Data ################
if clean_punk_data is True:
    # Status Update
    print('Cleaning Punk Data...')

    # Importing Data
    df_raw = pd.read_csv(f'punk_data/punk_data_raw.csv')
    df_clean = df_raw[['token_id', 'traits']]
    punks_traits = df_clean['traits'].tolist()
    shape = df_clean.shape

    # Setting up retrieval of accessories and types
    types = []
    accessories = []
    for punk in punks_traits:
        punk_dict = ast.literal_eval(punk)

        for trait in punk_dict:
            trait_type = trait['trait_type']
            trait_value = trait['value']

            if trait_type == 'type':
                types.append(trait_value)
            else:
                accessories.append(trait_value)

    # Filtering for unique results
    types_unique = list(set(types))
    accessories_unique = list(set(accessories))
    empty = [0] * shape[0]

    # Creating empty columns
    for accessory in accessories_unique:
        df_clean.insert(2, accessory, empty, True)
    for race in types_unique:
        df_clean.insert(2, race, empty, True)

    # Filling empty columns
    for trait_n in range(shape[0]):
        test = df_clean.iloc[trait_n, 1]
        test_punk_dict = ast.literal_eval(test)

        race = ''
        items = []
        for trait in test_punk_dict:
            trait_type = trait['trait_type']
            trait_value = trait['value']

            if trait_type == 'type':
                df_clean.at[trait_n, trait_value] = 1
            else:
                df_clean.at[trait_n, trait_value] = 1

    # Dropping Traits & Resetting Index
    df_clean = df_clean.sort_values('token_id')
    df_clean = df_clean.reset_index(drop=True)
    df_clean = df_clean.drop(columns=['traits'])

    # Saving as cleaned dataset
    df_clean.to_csv(f'punk_data/punk_data_clean.csv', index=False)
    print(f'Document saved as ==> punk_data/punk_data_clean.csv')

################ Get Punk Images ################
if get_punk_images is True:
    # Status Update
    print('Getting Punk Images...')
    # Download official image of all punks
    response = requests.get('https://raw.githubusercontent.com/larvalabs/cryptopunks/master/punks.png')
    file = open('punk_data/punks.png', 'wb')
    file.write(response.content)
    file.close()
    all_Punks = Image.open('punk_data/punks.png')

    # Splitting image in all different punks
    punk_number = 0
    for trait_n in range(100):
        for j in range(100):
            left = 24 * j
            right = left + 24
            top = 24 * trait_n
            bottom = top + 24
            punk = all_Punks.crop((left, top, right, bottom))
            punk.save(f'punk_images/{punk_number}.png')
            punk_number += 1

    print('Images saved in ==> punk_images/')


################ Data Builders ################
def training_data_builder(df_specific, trait_specific):
    # Oversampling Target to balance Classes
    target = 8000

    # Getting traits
    df_trait = df_specific[df_specific[trait_specific] == 1]

    # Getting no traits
    df_no_trait = df_specific[df_specific[trait_specific] == 0]

    # Oversampling
    df_trait_oversampled = df_trait.sample(n=target, replace=True).reset_index(drop=True)
    df_no_trait_oversampled = df_no_trait.sample(n=target, replace=True).reset_index(drop=True)

    # Putting dfs in a list
    df_oversampled = pd.concat([df_trait_oversampled, df_no_trait_oversampled])

    # Creating Training data
    items_training = df_oversampled['token_id'].tolist()
    training_data_trait_specific = []
    for k in items_training:
        punk_image_grey = cv2.imread(f'punk_images/{k}.png', cv2.IMREAD_GRAYSCALE)
        trait_value_training = int(df_specific[df_specific['token_id'] == k].loc[:, trait_specific])
        punk_attributes = np.eye(2)[trait_value_training]
        training_data_trait_specific.append([np.array(punk_image_grey) / 255, punk_attributes])

    training_data_trait_specific = np.array(training_data_trait_specific, dtype=object)
    np.random.shuffle(training_data_trait_specific)

    # Saving Training Data
    filename_training = f'punk_data/training_data/data_{trait_specific}.npy'
    np.save(filename_training, training_data_trait_specific)

    return training_data_trait_specific


def validation_data_builder(df_testing_data):
    # Creating Training data
    items_validation = df_testing_data['token_id'].tolist()
    testing_data_full = []
    for p in items_validation:
        punk_image_grey = cv2.imread(f'punk_images/{p}.png', cv2.IMREAD_GRAYSCALE)
        punk_attributes = df_testing_data[df_testing_data['token_id'] == p].iloc[:, 1:]
        testing_data_full.append([np.array(punk_image_grey) / 255, np.array(punk_attributes)[0]])

    testing_data_full = np.array(testing_data_full, dtype=object)

    # Saving Training Data
    np.save('punk_data/testing_data.npy', testing_data_full)

    return testing_data_full


################ Model Set-up ################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (2, 2))
        self.conv2 = nn.Conv2d(32, 64, (2, 2))
        self.conv3 = nn.Conv2d(64, 128, (2, 2))

        x = torch.randn(24, 24).view(-1, 1, 24, 24)
        self._to_linear = None
        self.convolutions(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convolutions(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


################ Forward Passing Data for in & out of sample Accuracy & Loss ################
def forward_pass(in_sample, network_trait, size, loss_function):
    # Sampling Data
    in_sample_sample = in_sample[np.random.choice(in_sample.shape[0], size, replace=False), :]
    in_sample_sample_X = torch.Tensor([point[0] for point in in_sample_sample]).view(-1, 24, 24).to(device)
    in_sample_sample_Y = torch.Tensor([point[1] for point in in_sample_sample]).to(device)

    # Calculate Accuracy
    predicted = network_trait(in_sample_sample_X.view(-1, 1, 24, 24)).detach()
    predicted_round = torch.round(predicted)
    matches = [torch.equal(pred, real) for pred, real in zip(predicted_round, in_sample_sample_Y)]
    accuracy = matches.count(True) / size

    # Calculate Loss
    loss = float(loss_function(predicted, in_sample_sample_Y))

    return [accuracy, loss]


def train_net(network_specific, training_data_specific, testing_data_specific, EPOCHS, batch_size):
    # Setting up Optimiser & loss function
    optimiser = optim.Adagrad(network_specific.parameters(), lr=0.01, lr_decay=0.01)
    loss_function = nn.MSELoss()

    # Putting in Tensor Format
    X_specific = torch.Tensor([item[0] for item in training_data_specific]).view(-1, 24, 24)
    y_specific = torch.Tensor([item[1] for item in training_data_specific])

    # Training Network
    all_accuracies_in = []
    all_accuracies_out = []
    all_losses_in = []
    all_losses_out = []
    for epoch in range(EPOCHS):
        for step in range(0, len(X_specific), batch_size):
            batch_X = X_specific[step:step + batch_size].view(-1, 1, 24, 24).to(device)
            batch_y = y_specific[step:step + batch_size].to(device)

            network_specific.zero_grad()
            outputs = network_specific(batch_X)

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimiser.step()

            # Testing In Sample Accuracy & Loss
            acc_in_step, loss_in_step = forward_pass(training_data_specific, network_specific,
                                                     batch_size, loss_function)

            # Testing Out of Sample Accuracy & Loss
            acc_out_step, loss_out_step = forward_pass(testing_data_specific, network_specific,
                                                       batch_size, loss_function)

            # Appending Data
            all_accuracies_in.append(acc_in_step)
            all_accuracies_out.append(acc_out_step)
            all_losses_in.append(loss_in_step)
            all_losses_out.append(loss_out_step)

    return [all_accuracies_in, all_accuracies_out, all_losses_in, all_losses_out]


################# Import Data, Splitting Data & Building Data #################
# Importing Necessary Items
df = pd.read_csv(f'punk_data/punk_data_clean.csv')
df_size = df.shape[0]
all_traits = df.columns[1:].tolist()
traits = all_traits

# Building all Data
if build_data is True:
    # Shuffling Data
    df = df.sample(frac=1).reset_index(drop=True)

    # Splitting Data
    train_data_n = int(train_fraction * df_size)
    training_data = df.iloc[0:train_data_n, :]
    testing_data = df.iloc[train_data_n:, :]

    # Building Training Data
    for trait in tq.tqdm(traits, desc='Building Data...'):
        training_data_builder(training_data, trait)

    # Building Training Data
    validation_data_builder(testing_data)

################# Network Training #################
if train_network is True:
    # Initialising lists
    networks = []
    accuracies_in = []
    accuracies_out = []
    losses_in = []
    losses_out = []

    # Iterating Over Traits
    for trait_n in tq.tqdm(range(len(traits)), desc='Training Networks...'):
        testing_data_built_trait = np.load('punk_data/testing_data.npy', allow_pickle=True)

        # Retrieving Specific Trait Testing Data
        trait = traits[trait_n]
        for test_item in range(len(testing_data_built_trait)):
            trait_number_value = testing_data_built_trait[test_item][1][trait_n]
            filtered = [1 - trait_number_value, trait_number_value]
            testing_data_built_trait[test_item][1] = filtered

        # Initialising Network
        net = Net().to(device)

        # Getting Trait Data
        trait_data = np.load(f'punk_data/training_data/data_{trait}.npy', allow_pickle=True)

        # Training Network
        acc_in, acc_out, loss_in, loss_out = train_net(net, trait_data, testing_data_built_trait, epo, 100)

        # Appending Network & losses
        networks.append(net)
        accuracies_in.append(acc_in)
        accuracies_out.append(acc_out)
        losses_in.append(loss_in)
        losses_out.append(loss_out)

    # Saving Results
    np.save('punk_data/networks/network.npy', networks)
    np.save('punk_data/networks/acc_loss_data/accuracies_in.npy', accuracies_in)
    np.save('punk_data/networks/acc_loss_data/accuracies_out.npy', accuracies_out)
    np.save('punk_data/networks/acc_loss_data/losses_in.npy', losses_in)
    np.save('punk_data/networks/acc_loss_data/losses_out.npy', losses_out)

################ Testing Data ################
if test_data is True:
    # Loading Test Data & Networks
    networks = np.load('punk_data/networks/network.npy', allow_pickle=True)
    testing_data_built = np.load('punk_data/testing_data.npy', allow_pickle=True)
    X = torch.Tensor([i[0] for i in testing_data_built]).view(-1, 24, 24).to(device)
    y = np.delete(testing_data_built, 0, 1)

    # Saving Testing Data Real Results in Comparable Format
    total = []
    for trait_n in range(len(y)):
        total.append(y[trait_n][0])
    y_real = pd.DataFrame(total)
    y_real.to_csv('punk_data/networks/y_real.csv', index=False)

    # Testing Data per Trait
    final = []
    for trait_n in tq.tqdm(range(len(y)), desc='Testing Networks on Test Data...'):
        out = []
        for network in networks:
            tensor_out = torch.round(network(X[trait_n].view(-1, 1, 24, 24))[0]).cpu()
            value_of_trait = tensor_out.detach().numpy()[1].astype(int)
            out.append(np.array(value_of_trait))
        final.append(out)
    y_pred = pd.DataFrame(final)

    # Saving Testing Data Predicted Results in Comparable Format
    y_pred.to_csv('punk_data/networks/y_pred.csv', index=False)

################ Evaluate Performance ################
if evaluate_performance is True:
    # Importing Data
    y_pred = pd.read_csv('punk_data/networks/y_pred.csv')
    y_real = pd.read_csv('punk_data/networks/y_real.csv')

    # Calculating Performance
    performance = []
    for i in range(len(traits)):
        # Calculating Metrics per trait
        tn, fp, fn, tp = confusion_matrix(y_real.iloc[:, i].values, y_pred.iloc[:, i].values).ravel()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fnr = fn / (fn + tp)
        tnr = tn / (tn + fp)
        acc_trait = accuracy_score(y_real.iloc[:, i].values, y_pred.iloc[:, i].values)
        pre_trait = precision_score(y_real.iloc[:, i].values, y_pred.iloc[:, i].values, zero_division=1)
        rec_trait = recall_score(y_real.iloc[:, i].values, y_pred.iloc[:, i].values)
        f1_trait = f1_score(y_real.iloc[:, i].values, y_pred.iloc[:, i].values)
        performance.append([traits[i], tn, fp, fn, tp, fpr, tpr, fnr, tnr, acc_trait, pre_trait, rec_trait, f1_trait])

    # Converting & Saving Results to csv
    df_performance = pd.DataFrame(performance, columns=['Trait', 'tn', 'fp', 'fn', 'tp', 'fpr', 'tpr', 'fnr', 'tnr',
                                                        'Accuracy', 'Precision', 'Recall', 'F1'])

    # Saving Performance Results
    df_performance.to_csv('punk_data/networks/performance.csv', index=False)

################ Plotting Results ################
if plot_results is True:
    # Importing Data
    acc_in = np.load('punk_data/networks/acc_loss_data/accuracies_in.npy', allow_pickle=True)
    acc_out = np.load('punk_data/networks/acc_loss_data/accuracies_out.npy', allow_pickle=True)
    loss_in = np.load('punk_data/networks/acc_loss_data/losses_in.npy', allow_pickle=True)
    loss_out = np.load('punk_data/networks/acc_loss_data/losses_out.npy', allow_pickle=True)
    acc_in_epo_one = acc_in[:, :160]
    acc_out_epo_one = acc_out[:, :160]
    loss_in_epo_one = loss_in[:, :160]
    loss_out_epo_one = loss_out[:, :160]
    df_performance = pd.read_csv('punk_data/networks/performance.csv')

    # Plotting All Accuracies & Losses out of Sample
    all_curves, (acc, los) = plt.subplots(2, 1, sharex=True)
    all_curves.suptitle('All Accuracies & Losses out of Sample', fontsize=14)
    acc.plot(acc_out.T, label='Accuracy out of Sample', linewidth=1)
    acc.set_title('For Twelve EPOCH')
    acc.set_ylabel('Accuracy')
    acc.set_ylim(bottom=0, top=1)
    los.plot(loss_out.T, label='Loss out of Sample', linewidth=1)
    los.set_ylabel('Loss')
    los.set_xlabel('Steps')
    los.set_ylim(bottom=0, top=1)
    all_curves.savefig('punk_data/networks/graphs/all_out_evolution.png')
    plt.clf()

    # Averaging out Accuracies & Loss
    acc_in = acc_in.sum(axis=0) / 92
    acc_out = acc_out.sum(axis=0) / 92
    loss_in = loss_in.sum(axis=0) / 92
    loss_out = loss_out.sum(axis=0) / 92

    # Plotting Average Accuracies & Losses in and out of Sample
    avg_curves, (acc, los) = plt.subplots(2, 1, sharex=True)
    avg_curves.suptitle('Average Accuracy & Loss Evolution in and out of Sample', fontsize=14)
    acc.plot(acc_in, label='Accuracy in Sample')
    acc.plot(acc_out, label='Accuracy out of Sample')
    acc.set_title('For Twelve EPOCH')
    acc.set_ylabel('Accuracy')
    acc.set_ylim(bottom=0, top=1)
    acc.legend(loc='lower left')
    los.plot(loss_in, label='Loss in Sample')
    los.plot(loss_out, label='Loss out of Sample')
    los.set_ylabel('Loss')
    los.set_xlabel('Steps')
    los.set_ylim(bottom=0, top=1)
    los.legend(loc='upper left')
    avg_curves.savefig('punk_data/networks/graphs/avg_in_out_evolution.png')
    plt.clf()

    # Plotting All Accuracies & Losses out of Sample for One EPOCH
    all_curves_epo_one, (acc, los) = plt.subplots(2, 1, sharex=True)
    all_curves_epo_one.suptitle('All Accuracies & Losses out of Sample', fontsize=14)
    acc.plot(acc_in_epo_one.T, label='Accuracy out of Sample', linewidth=1)
    acc.set_ylabel('Accuracy')
    acc.set_ylim(bottom=0, top=1)
    los.plot(loss_out_epo_one.T, label='Loss out of Sample', linewidth=1)
    los.set_ylabel('Loss')
    los.set_xlabel('Steps')
    los.set_ylim(bottom=0, top=1)
    acc.set_title('For One EPOCH')
    all_curves_epo_one.savefig('punk_data/networks/graphs/all_out_evolution_epo_one.png')
    plt.clf()

    # Averaging out Accuracies & Loss for One EPOCH
    acc_in_epo_one = acc_in_epo_one.sum(axis=0) / 92
    acc_out_epo_one = acc_out_epo_one.sum(axis=0) / 92
    loss_in_epo_one = loss_in_epo_one.sum(axis=0) / 92
    loss_out_epo_one = loss_out_epo_one.sum(axis=0) / 92

    # Plotting Average Accuracies & Losses in and out of Sample for One EPOCH
    avg_curves_epo_one, (acc, los) = plt.subplots(2, 1, sharex=True)
    avg_curves_epo_one.suptitle('Average Accuracy & Loss Evolution in and out of Sample', fontsize=14)
    acc.plot(acc_in_epo_one, label='Accuracy in Sample')
    acc.plot(acc_out_epo_one, label='Accuracy out of Sample')
    acc.set_title('For One EPOCH')
    acc.set_ylabel('Accuracy')
    acc.set_ylim(bottom=0, top=1)
    acc.legend(loc='lower left')
    los.plot(loss_in_epo_one, label='Loss in Sample')
    los.plot(loss_out_epo_one, label='Loss out of Sample')
    los.set_ylabel('Loss')
    los.set_xlabel('Steps')
    los.set_ylim(bottom=0, top=1)
    los.legend(loc='upper left')
    avg_curves_epo_one.savefig('punk_data/networks/graphs/avg_in_out_evolution_epo_one.png')
    plt.clf()

    # Plotting Cumulative Performances
    fig = plt.figure(1, (7, 6))
    ax = fig.add_subplot(1, 1, 1)
    columns = df_performance.columns.tolist()[9:]
    for i in columns:
        temp = df_performance[i].values
        sorted_data = np.sort(temp)
        ax.step(sorted_data, np.arange(sorted_data.size) / 0.92, label=i)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel('Metric Performance')
    ax.set_ylabel('Percentage of Networks')
    ax.legend()
    fig.suptitle('Cumulative Distributions of Performances', size=14)
    fig.savefig('punk_data/networks/graphs/cumulative_performance.png')
    plt.clf()

################ Creating Archive ################
if create_archive is True:
    shutil.make_archive('cryptopunks_analysis', 'zip', os.getcwd())

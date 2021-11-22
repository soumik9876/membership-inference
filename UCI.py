import keras
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

LEARNING_RATE = 0.001
EPOCH = 100
TRAINING_SIZE = 2700
TEST_SIZE = 1500
NUM_TARGET = 1
NUM_SHADOW = 20
IN = 1
OUT = 0
VERBOSE = 1


def load_data(path='UCI.adult.data'):
    from keras.utils.data_utils import get_file

    path = get_file(path, origin='http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    f = pd.read_csv(path, header=None)

    # encode the categorical values
    for col in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        le = LabelEncoder()
        f[col] = le.fit_transform(f[col].astype('str'))

    # normalize the values
    x_range = [i for i in range(14)]
    f[x_range] = f[x_range] / f[x_range].max()

    x = f[x_range].values
    y = f[14].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
    return (x_train, y_train), (x_test, y_test)


def load_senti_data(data):
    dataframe = pd.read_csv(f"sent_nob/{data}", header=None)
    le = LabelEncoder()
    for col in range(2):
        dataframe[col] = le.fit_transform(dataframe[col].astype('str'))
    x_range = [i for i in range(2)]
    dataframe[x_range] = dataframe[x_range] / dataframe[x_range].max()

    x = dataframe[0].values
    y = dataframe[1].values
    print(x, y)
    return x, y


def load_hate_data():
    dataframe = pd.read_csv(f"hate_speech_data/Bengali_hate_speech.csv", header=None, encoding="utf-8")
    le = LabelEncoder()
    for col in range(2):
        dataframe[col] = le.fit_transform(dataframe[col].astype('str'))
    x_range = [i for i in range(2)]
    dataframe[x_range] = dataframe[x_range] / dataframe[x_range].max()

    x = dataframe[0].values
    y = dataframe[1].values
    print(x, y)
    return x, y


def load_hate_data_optimized():
    # dataframe = pd.read_csv(f"hate_speech_data/Bengali_hate_speech.csv", header=None)

    path = 'hate_speech_data/Bengali_hate_speech.csv'
    dataframe = pd.read_csv(path, encoding='utf-8')

    dataframe = dataframe.sample(frac=1,
                                 random_state=42)  # shufling the dataset, random state = 42 ensures reproducibility.

    # remiving everything except Bengali text
    dataframe['sentence'] = dataframe['sentence'].str.replace(r'[^\u0980-\u09FF ]+', ' ')

    # droppig duplicates
    dataframe.dropna(subset=['sentence'], inplace=True)

    # removing empty rows
    dataframe.drop_duplicates(subset=['sentence'], inplace=True)

    # le = LabelEncoder()
    # for col in range(2):
    #     dataframe[col] = le.fit_transform(dataframe[col].astype('str'))
    # x_range = [i for i in range(2)]
    # dataframe[x_range] = dataframe[x_range] / dataframe[x_range].max()

    # x = dataframe[0].values
    # y = dataframe[1].values

    max_features = 80000  # Needs to define optimal one later
    tokenizer = Tokenizer(num_words=max_features, split=' ')

    tokenizer.fit_on_texts(dataframe['sentence'].values)
    X = tokenizer.texts_to_sequences(dataframe['sentence'].values)
    X = pad_sequences(X)

    Y = pd.get_dummies(dataframe['hate']).values
    print(X, Y)
    return X, Y


def sample_data(train_data, test_data, num_sets):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    new_x_train, new_y_train = [], []
    new_x_test, new_y_test = [], []
    for i in range(num_sets):
        x_temp, y_temp = resample(x_train, y_train, n_samples=TRAINING_SIZE, random_state=0)
        new_x_train.append(x_temp)
        new_y_train.append(y_temp)
        print(TEST_SIZE)
        x_temp, y_temp = resample(x_test, y_test, n_samples=TEST_SIZE, random_state=0)
        new_x_test.append(x_temp)
        new_y_test.append(y_temp)
    return (new_x_train, new_y_train), (new_x_test, new_y_test)


def build_fcnn_model():
    from keras.models import Sequential
    from keras.layers import Dense
    # build the model
    model = Sequential()
    model.add(Dense(512, input_dim=534, activation='tanh'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(2, activation='sigmoid'))

    model.summary()
    return model


def get_trained_keras_models(keras_model, train_data, test_data, num_models):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    models = []
    for i in range(num_models):
        models.append(keras.models.clone_model(keras_model))
        rms = keras.optimizers.RMSprop(lr=LEARNING_RATE, decay=1e-7)
        models[i].compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
        models[i].fit(x_train[i], y_train[i], batch_size=32, epochs=EPOCH, verbose=VERBOSE, shuffle=True)
        score = models[i].evaluate(x_test[i], y_test[i], verbose=VERBOSE)
        print('\n', 'Model ', i, ' test accuracy:', score[1])
    return models


def get_attack_dataset(models, train_data, test_data, num_models, data_size):
    # generate dataset for the attack model
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    num_classes = 1
    x_data, y_data = [[] for i in range(num_classes)], [[] for i in range(num_classes)]
    for i in range(num_models):
        # IN data
        x_temp, y_temp = resample(x_train[i], y_train[i], n_samples=data_size, random_state=0)
        for j in range(data_size):
            y_idx = np.argmax(y_temp[j]) - 1
            print(f'j is {j}', len(x_temp), x_temp.shape)
            try:
                x_data[y_idx].append(models[i].predict(x_temp[j:j + 1])[0])
            except Exception as e:
                print(e)
            y_data[y_idx].append(IN)
        # OUT data
        x_temp, y_temp = resample(x_test[i], y_test[i], n_samples=data_size, random_state=0)
        for j in range(data_size):
            y_idx = np.argmax(y_temp[j]) - 1
            print(f'y_idx : {y_idx} {len(x_data)} {len(y_data)}')
            x_data[y_idx].append(models[i].predict(x_temp[j:j + 1])[0])
            y_data[y_idx].append(OUT)
    return x_data, y_data


def get_trained_svm_models(train_data, test_data, num_models=1):
    from sklearn import svm
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    models = []
    for i in range(num_models):
        print('Training svm model : ', i)
        models.append(svm.SVC(gamma='scale', kernel='linear', verbose=VERBOSE))
        models[i].fit(x_train[i], y_train[i])
        score = models[i].score(x_test[i], y_test[i])
        print('SVM model ', i, 'score : ', score)
    return models


def senti_data():
    x_train, y_train = load_senti_data('Train.csv')
    x_test, y_test = load_senti_data('Test.csv')
    # split the data for each model
    return x_train, y_train, x_test, y_test


def hate_data():
    x, y = load_hate_data_optimized()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = hate_data()
    print(x_train.shape, y_train.shape)
    # (20843, 534) (20843, 2)

    # uncomment to train senti data
    # x_train, y_train, x_test, y_test = senti_data()

    target_train = (x_train[:TRAINING_SIZE * NUM_TARGET], y_train[:TRAINING_SIZE * NUM_TARGET])
    target_test = (x_test[:TEST_SIZE * NUM_TARGET], y_test[:TEST_SIZE * NUM_TARGET])
    target_train_data, target_test_data = sample_data(target_train, target_test, NUM_TARGET)

    shadow_train = (x_train[TRAINING_SIZE * NUM_TARGET:], y_train[TRAINING_SIZE * NUM_TARGET:])
    shadow_test = (x_test[TEST_SIZE * NUM_TARGET:], y_test[TEST_SIZE * NUM_TARGET:])
    shadow_train_data, shadow_test_data = sample_data(shadow_train, shadow_test, NUM_SHADOW)

    cnn_model = build_fcnn_model()
    # compile the target model
    target_models = get_trained_keras_models(cnn_model, target_train_data, target_test_data, NUM_TARGET)
    # compile the shadow models
    shadow_models = get_trained_keras_models(cnn_model, shadow_train_data, shadow_test_data, NUM_SHADOW)

    # get train data for the attack model
    attack_train = get_attack_dataset(shadow_models, shadow_train_data, shadow_test_data, NUM_SHADOW, TEST_SIZE)
    # get test data for the attack model
    attack_test = get_attack_dataset(target_models, target_train_data, target_test_data, NUM_TARGET, TEST_SIZE)

    # training the attack model
    attack_model = get_trained_svm_models(attack_train, attack_test)

    # TODO generate the report


if __name__ == '__main__':
    main()

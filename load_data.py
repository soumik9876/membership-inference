def load_hate_data():
    # dataframe = pd.read_csv(f"hate_speech_data/Bengali_hate_speech.csv", header=None)

    path = '/content/drive/MyDrive/CMPUT622/CMPUT-622_project/Hate Speech Data/Bengali_hate_speech.csv'
    dataframe = pd.read_csv(path,encoding='utf-8')

    dataframe = dataframe.sample(frac=1,random_state=42) #shufling the dataset, random state = 42 ensures reproducibility.

    # remiving everything except Bengali text
    dataframe['sentence'] = dataframe['sentence'].str.replace(r'[^\u0980-\u09FF ]+', ' ')

    # droppig duplicates
    dataframe.dropna(subset=['sentence'],inplace=True)

    # removing empty rows
    dataframe.drop_duplicates(subset=['sentence'],inplace=True)

    # le = LabelEncoder()
    # for col in range(2):
    #     dataframe[col] = le.fit_transform(dataframe[col].astype('str'))
    # x_range = [i for i in range(2)]
    # dataframe[x_range] = dataframe[x_range] / dataframe[x_range].max()

    # x = dataframe[0].values
    # y = dataframe[1].values

    max_features = 80000 # Needs to define optimal one later
    tokenizer = Tokenizer(num_words=max_features, split=' ')

    tokenizer.fit_on_texts(dataframe['sentence'].values)
    X = tokenizer.texts_to_sequences(dataframe['sentence'].values)
    X = pad_sequences(X)

    Y = pd.get_dummies(dataframe['hate']).values
    print(X, Y)
    return X, Y
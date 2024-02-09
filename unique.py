import pandas as pd

def identify_unique_sentiments(dataset_path, output_file):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Get the unique sentiments
    unique_sentiments = set(df['sentiment'])

    # Save unique sentiments to a text file
    with open(output_file, 'w') as file:
        for sentiment in unique_sentiments:
            file.write(sentiment + '\n')
    
    print("Unique sentiments saved to", output_file)

dataset_path = "data.csv"
output_file = "unique_sentiments.txt"
identify_unique_sentiments(dataset_path, output_file)
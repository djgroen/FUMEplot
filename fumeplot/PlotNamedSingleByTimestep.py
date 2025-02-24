import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    #df = pd.read_csv("../sample_homecoming_agentlog/1/migration.log")

    # Count occurrences of each source
    #source_counts = df['source'].value_counts()
    
    # Plot histogram
    #plt.figure(figsize=(10, 6))
    #sns.barplot(x=source_counts.index, y=source_counts.values, palette='Set3')
    #plt.bar(source_counts.index, source_counts.values, color='skyblue')

    # Read and aggregate data from multiple CSV files
    all_counts = []
    
    csv_files = []
    for i in range(1,11):
        csv_files.append(f"../sample_homecoming_agentlog/{i}/migration.log")

    for file in csv_files:
        df = pd.read_csv(file)
        source_counts = df['source'].value_counts()
        all_counts.append(source_counts)
    
    # Combine counts into a DataFrame, filling missing values with 0
    combined_counts = pd.DataFrame(all_counts).fillna(0)
    
    # Compute mean and standard deviation for each source
    mean_counts = combined_counts.mean()
    std_counts = combined_counts.std()
    
    # Plot histogram with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(mean_counts.index, mean_counts.values, yerr=std_counts.values, color='skyblue', capsize=5)

    # Labels and title
    plt.xlabel('Source Country')
    plt.ylabel('Number of Entries')
    plt.title('Histogram of Entries Grouped by Source')
    plt.xticks(rotation=45)
    
    # Show plot
    plt.show()


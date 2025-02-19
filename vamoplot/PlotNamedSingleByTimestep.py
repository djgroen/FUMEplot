import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    df = pd.read_csv("../sample_homecoming_agentlog/1/migration.log")

    # Count occurrences of each source
    source_counts = df['source'].value_counts()
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    #sns.barplot(x=source_counts.index, y=source_counts.values, palette='Set3')
    plt.bar(source_counts.index, source_counts.values, color='skyblue')

    # Labels and title
    plt.xlabel('Source Country')
    plt.ylabel('Number of Entries')
    plt.title('Histogram of Entries Grouped by Source')
    plt.xticks(rotation=45)
    
    # Show plot
    plt.show()


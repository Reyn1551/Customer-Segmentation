# --- Function Definitions ---

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Define features
    fitur_numerik = [
        'Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer',
        'Num_of_Purchases', 'Average_Transaction_Amount', 'Num_of_Returns',
        'Num_of_Support_Contacts', 'Satisfaction_Score', 'Last_Purchase_Days_Ago'
    ]
    
    fitur_kategorikal = [
        'Gender', 'Email_Opt_In', 'Promotion_Response', 'Target_Churn'
    ]
    
    # Scale numeric features
    scaler = StandardScaler()
    df_scaled_numerik = scaler.fit_transform(df[fitur_numerik])
    df_scaled_numerik = pd.DataFrame(df_scaled_numerik, columns=fitur_numerik, index=df.index)

    # One-Hot Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_encoded_kategorikal = encoder.fit_transform(df[fitur_kategorikal])
    df_encoded_kategorikal = pd.DataFrame(df_encoded_kategorikal, columns=encoder.get_feature_names_out(fitur_kategorikal), index=df.index)

    # Combine processed features
    df_processed = pd.concat([df_scaled_numerik, df_encoded_kategorikal], axis=1)
    return df_processed

def determine_optimal_clusters(df_processed):
    inertia = []
    silhouette_scores = []
    range_n_clusters = range(2, 11)

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(df_processed)
        inertia.append(kmeans.inertia_)
        if n_clusters > 1:
            silhouette_scores.append(silhouette_score(df_processed, kmeans.labels_))

    return inertia, silhouette_scores, range_n_clusters

def cluster_data(df_processed, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_processed)
    return df

# --- Streamlit App ---

st.title("Customer Segmentation Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data Loaded Successfully!")
    st.dataframe(df.head())

    # Preprocess data
    df_processed = preprocess_data(df)
    st.write("--- Data After Preprocessing ---")
    st.dataframe(df_processed.head())

    # Determine optimal clusters
    inertia, silhouette_scores, range_n_clusters = determine_optimal_clusters(df_processed)

    # Plot Elbow Method
    st.subheader("Elbow Method for Optimal K")
    fig, ax = plt.subplots()
    ax.plot(range_n_clusters, inertia, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)

    # Plot Silhouette Scores
    if silhouette_scores:
        st.subheader("Silhouette Scores for Optimal K")
        fig, ax = plt.subplots()
        ax.plot(list(range_n_clusters)[1:], silhouette_scores, marker='o', color='red')
        ax.set_title('Silhouette Scores')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Silhouette Score')
        st.pyplot(fig)

    # User input for number of clusters
    n_clusters_chosen = st.number_input("Choose the number of clusters (K)", min_value=2, max_value=10, value=3)

    # Cluster data
    clustered_data = cluster_data(df_processed, n_clusters_chosen)
    st.write("--- Clustering Results ---")
    st.dataframe(clustered_data.head())

    # Show cluster distribution
    st.write("Cluster Distribution:")
    st.bar_chart(clustered_data['Cluster'].value_counts())

    # Further analysis and profiling can be added here...

# Run the app with: streamlit run customer_segmentation_app.py
```

### Explanation of the Code:
1. **File Upload**: The app allows users to upload a CSV file containing customer data.
2. **Data Preprocessing**: The uploaded data is preprocessed (scaling and encoding).
3. **Optimal Clustering**: The app calculates the optimal number of clusters using the Elbow method and silhouette scores.
4. **User Input**: Users can select the number of clusters they want to use for K-Means clustering.
5. **Clustering Results**: The app displays the clustered data and a bar chart showing the distribution of customers across clusters.

### Running the App:
To run the Streamlit app, navigate to the directory where your `customer_segmentation_app.py` file is located and run:

```bash
streamlit run customer_segmentation_app.py
```

This will start a local server, and you can view the app in your web browser at `http://localhost:8501`. 

### Further Enhancements:
You can expand the app by adding more features such as:
- Detailed profiling of each cluster.
- Recommendations based on cluster characteristics.
- Interactive visualizations for better insights.
- Exporting results to a new CSV file. 

Feel free to customize the app according to your needs!
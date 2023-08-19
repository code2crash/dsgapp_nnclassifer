import streamlit as st
import io
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


model = None

if not model:
    st.write("Format: 1st. row column labels. One column per variable, including the dependent variable. Values have to be encoded in a numerical form. Example with 3 variables (columns) and 3 samples:")
    df_styler = pd.DataFrame({'Variable 1': [0.0, 4.1, 2.3], 'Variable 2': [0, 1, 1], 'Variable 3': [5, 3, 9]}).style.hide()
    st.table(data=df_styler)

    train_file = st.file_uploader("Choose a file")
    if train_file is not None:
    
        # Can be used wherever a "file-like" object is accepted:
        df_train = pd.read_csv(train_file, delimiter=';')

        # select dependent variable
        target_col = st.selectbox('Select dependent variable',
                              df_train.columns
                             )

        dependent_cols = df_train.drop(target_col, axis=1).columns

        n_neighbors = st.number_input("Number of neighbours to use", min_value=0, format='%i', value=3) 

        # train model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(df_train.drop(target_col, axis=1), df_train[[target_col]])
        
if not model:
    status = ":red[MODEL NOT TRAINED]"
else:
    status = ":green[MODEL TRAINED]"

st.markdown(status)

if model:
    st.write("Upload a csv file containing samples to generate proedictions for")
    test_file = st.file_uploader("Choose a file", key='test_file_uploader')
    if test_file is not None:
        df_test = pd.read_csv(test_file, delimiter=';', usecols=dependent_cols)[dependent_cols]

        y_hat = model.predict(df_test.values)

        df_y_hat = pd.DataFrame({target_col: y_hat}) # pd.concat([df_test, pd.DataFrame({target_col: y_hat})], axis=1)

        # csv = bla.to_csv(sep=';', index=False, encoding='utf-8')

        # xlsx = bla.to_excel('predictions.xlsx', sheet_name='predictions')

        (par_keys, par_values) = zip(*model.get_params().items())
        df_pars = pd.DataFrame({'Model Parameters': par_keys, 'Value': par_values})
        

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:  
            df_y_hat.to_excel(writer, sheet_name='Predictions', index=False)
            df_test.to_excel(writer, sheet_name='Samples for Predictions', index=False)
            df_train.to_excel(writer, sheet_name='Training Samples', index=False)
            df_pars.to_excel(writer, sheet_name='Model parameters', index=False)


        st.download_button(
            label="Download predictions as xlsx",
            data=buffer,
            file_name='predictions.xlsx',
            mime='application/vnd.ms-excel',
        )
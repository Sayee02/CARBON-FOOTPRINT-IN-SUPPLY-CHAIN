import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import os
import math

@st.cache_resource
def load_model(path='model.joblib'):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_sample_features(path='X_train.csv'):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def main():
    st.set_page_config(page_title='Carbon Emission Predictor', layout='wide')
    st.title('Carbon Emission Optimization — Prediction UI')
    st.write('Use below categories to predict carbon emissions')
    st.write('Road: 0, Rail: 1, Sea: 2, Air: 3')
    st.write('Petrol: 0, Diesel: 1, CNG: 2, Electric: 3')
    st.write('North: 2, South: 0, East: 1, West: 3')
    
    # ---- Units mapping: simple heuristics to attach units to feature names ----
    units_map = {
        'route_distance': 'km',
        'distance': 'km',
        'km': 'km',
        'weight': 'kg',
        'kg': 'kg',
        'transport_mode': '',
        'fuel_type': '',
        'fuel_consumption': 'L',
        'fuel_consumption_l': 'L',
        'region': '',
        'delivery_time': 'days',
        'time_days': 'days',
        'cost': 'USD',
        'usd': 'USD',
        'high_emission_flag': '',
        'fuel_efficiency': '',
        'cost_efficiency': '',
        'emission': 'kgCO2',
        'carbon': 'kgCO2'
    }

    def get_unit(col_name: str) -> str:
        low = col_name.lower()
        for key, unit in units_map.items():
            if key in low:
                return unit
        return ''

    # Sidebar controls for tree calculation and CO2 emission unit
    st.sidebar.header('Offset / Tree settings')
    co2_emission_unit = st.sidebar.selectbox('CO2 Emission unit', ['kgCO2', 'tCO2', 'custom'], index=0)
    custom_to_kg = 1.0
    if co2_emission_unit == 'tCO2':
        emission_to_kg = 1000.0
    elif co2_emission_unit == 'kgCO2':
        emission_to_kg = 1.0
    else:
        custom_text = st.sidebar.text_input('Custom unit (value -> kgCO2 factor)', '1.0')
        try:
            emission_to_kg = float(custom_text)
        except Exception:
            emission_to_kg = 1.0

    sequestration_per_tree = st.sidebar.number_input('Sequestration per tree (kg CO2 / year)', value=21.77)
    sequestration_years = st.sidebar.number_input('Years trees will sequester (years)', value=20)

    model = load_model('model.joblib')
    if model is None:
        st.error('Model file `model.joblib` not found in the project directory.')
        st.stop()

    sample_df = load_sample_features('X_train.csv')
    if sample_df is None:
        st.warning('`X_train.csv` not found — you can upload a CSV in Batch mode or enter values manually.')

    st.sidebar.header('Prediction Mode')
    mode = st.sidebar.selectbox('Mode', ['Single input', 'Batch upload'])

    # Define output names (assuming model predicts in this order)
    output_names = ['high_emission_flag', 'fuel_efficiency', 'cost_efficiency', 'co2_emission']

    # Determine feature names from sample_df if available, else allow user to provide/upload
    if sample_df is not None:
        feature_names = list(sample_df.columns)
    else:
        feature_names = None

    if mode == 'Single input':
        st.header('Single prediction')
        st.write('Enter feature values to predict for a single shipment/route.')

        user_input = {}
        if feature_names is not None:
            for col in feature_names:
                # Choose sensible defaults from sample_df if possible
                col_series = sample_df[col]
                unit = get_unit(col)
                label = f"{col} ({unit})" if unit else col
                if pd.api.types.is_numeric_dtype(col_series):
                    default = float(col_series.median())
                    user_input[col] = st.number_input(label, value=default, format='%f')
                else:
                    # treat as string/categorical
                    user_input[col] = st.text_input(label, value=str(col_series.mode().iat[0]) if not col_series.mode().empty else '')
        else:
            st.write('No feature list found. Upload a sample CSV or enter comma-separated feature names below.')
            cols_text = st.text_input('Feature names (comma-separated)', '')
            if cols_text:
                feature_names = [c.strip() for c in cols_text.split(',') if c.strip()]
                for col in feature_names:
                    user_input[col] = st.text_input(col)

        if st.button('Predict single'):
            if not feature_names:
                st.error('Feature names unknown. Provide a sample or enter names first.')
            else:
                input_df = pd.DataFrame([user_input], columns=feature_names)
                try:
                    preds = model.predict(input_df)
                    # Assume preds is (1, 4) for single prediction
                    pred_values = np.asarray(preds).ravel()  # Flatten to 1D array of 4 values
                    if len(pred_values) != 4:
                        st.error('Model does not output 4 predictions as expected.')
                    else:
                        st.success('Predictions:')
                        for i, name in enumerate(output_names):
                            if name == 'co2_emission':
                                st.write(f'{name}: {pred_values[i]} {co2_emission_unit}')
                            else:
                                st.write(f'{name}: {pred_values[i]}')
                        
                        # Calculate trees needed to offset predicted CO2 emission (converted to kgCO2)
                        co2_kg = pred_values[3] * emission_to_kg
                        per_tree_total = sequestration_per_tree * sequestration_years
                        if per_tree_total > 0:
                            trees_needed = math.ceil(co2_kg / per_tree_total)
                        else:
                            trees_needed = None
                        if trees_needed is not None:
                            st.info(f'Estimated trees to plant to offset this CO2 emission: {trees_needed} trees \n(assumes {sequestration_per_tree} kgCO2/year per tree for {sequestration_years} years)')
                        st.write('Input')
                        st.dataframe(input_df)
                except Exception as e:
                    st.error(f'Prediction failed: {e}')

    else:
        st.header('Batch prediction')
        st.write('Upload a CSV (same features used by the model). Or use the included `X_test.csv` if present.')
        uploaded = st.file_uploader('Upload CSV', type=['csv'])
        use_sample = st.button('Use included `X_test.csv`')

        batch_df = None
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f'Failed to read uploaded CSV: {e}')
        elif use_sample:
            sample_path = 'X_test.csv'
            if os.path.exists(sample_path):
                batch_df = pd.read_csv(sample_path)
            else:
                st.error('`X_test.csv` not found in project directory.')

        if batch_df is not None:
            st.write('Preview of uploaded data:')
            st.dataframe(batch_df.head())

            if feature_names is not None:
                # Ensure columns align — reindex or warn
                missing = [c for c in feature_names if c not in batch_df.columns]
                if missing:
                    st.warning(f'Missing columns in uploaded file: {missing}. Attempting to predict with available columns.')
                # Reindex to feature_names where possible
                try:
                    input_for_model = batch_df.reindex(columns=feature_names)
                except Exception:
                    input_for_model = batch_df
            else:
                input_for_model = batch_df

            if st.button('Run batch prediction'):
                try:
                    preds = model.predict(input_for_model.fillna(0))
                    preds = np.asarray(preds)  # Ensure it's an array
                    if preds.ndim == 1:
                        preds = preds.reshape(-1, 1)  # If single output, but we expect 4
                    if preds.shape[1] != 4:
                        st.error('Model does not output 4 predictions as expected.')
                    else:
                        out = input_for_model.copy()
                        for i, name in enumerate(output_names):
                            out[name] = preds[:, i]
                        
                        # Calculate trees needed per row based on CO2 emission
                        co2_kg_arr = preds[:, 3] * emission_to_kg
                        per_tree_total = sequestration_per_tree * sequestration_years
                        if per_tree_total > 0:
                            trees_arr = [math.ceil(x / per_tree_total) for x in co2_kg_arr]
                        else:
                            trees_arr = [None] * len(co2_kg_arr)
                        out['trees_needed'] = trees_arr
                        st.success('Batch prediction complete')
                        st.dataframe(out.head())

                        # Download button
                        csv_bytes = out.to_csv(index=False).encode('utf-8')
                        st.download_button('Download predictions CSV', data=csv_bytes, file_name='predictions.csv', mime='text/csv')
                except Exception as e:
                    st.error(f'Batch prediction failed: {e}')

    # Optional: feature importance if available
    st.sidebar.header('Model info')
    if hasattr(model, 'feature_importances_') and sample_df is not None:
        try:
            fi = model.feature_importances_
            fi_df = pd.DataFrame({'feature': sample_df.columns, 'importance': fi})
            fi_df = fi_df.sort_values('importance', ascending=False)
            st.sidebar.write('Feature importances')
            st.sidebar.bar_chart(fi_df.set_index('feature'))
        except Exception:
            st.sidebar.write('Could not show feature importances for this model.')

if __name__ == '__main__':
    main()

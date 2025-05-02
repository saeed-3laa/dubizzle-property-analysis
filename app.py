import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import numpy as np
import re
from typing import Dict, Any, List, Tuple

# Set page configuration
st.set_page_config(
    page_title="🏡 Dubizzle Egypt Property Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2c3e50;
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    .fun-fact {
        background-color: #3498db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.filtered_data = None
    st.session_state.metrics = None
    st.session_state.df = None

# Load and preprocess data with caching
@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv('Dubizzle_properties.csv')
        
        # Replace "Not Available" and "Error" with NA
        df.replace(["Not Available", "Error"], pd.NA, inplace=True)
        
        # Drop rows where all columns (except first) are NA
        df.dropna(subset=df.columns[1:], how="all", inplace=True)
        
        # Clean numerical columns
        df['price'] = pd.to_numeric(df['price'].apply(lambda x: re.sub(r'EGP\s*([\d,]+)', r'\1', str(x)).replace(',', '')), errors='coerce')
        df['down_payment'] = pd.to_numeric(df['down_payment'].apply(lambda x: re.sub(r'[^\d]', '', str(x)) if "0%" not in str(x) else '0'), errors='coerce')
        df['area'] = pd.to_numeric(df['area'].apply(lambda x: re.sub(r'(\d+)\s*m²', r'\1', str(x))), errors='coerce')
        df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
        df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
        
        # Parse creation date
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            match = re.search(r'(\d+)', str(date_str))
            if not match:
                return pd.NaT
            num = int(match.group(1))
            date_str = str(date_str).lower()
            if "day" in date_str:
                return datetime.now() - timedelta(days=num)
            elif "hour" in date_str:
                return datetime.now() - timedelta(hours=num)
            elif "minute" in date_str:
                return datetime.now() - timedelta(minutes=num)
            elif "week" in date_str:
                return datetime.now() - timedelta(weeks=num)
            elif "month" in date_str:
                return datetime.now() - timedelta(days=30*num)
            else:
                return pd.NaT
        
        df['creation_date'] = df['creation_date'].apply(parse_date)
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
        
        # Split location into area_name and city
        def split_location_regex(loc_str):
            if pd.isna(loc_str):
                return pd.Series([None, None])
            match = re.match(r'^(.*?),\s*(.*)$', str(loc_str))
            if match:
                return pd.Series([match.group(1).strip(), match.group(2).strip()])
            else:
                return pd.Series([loc_str.strip(), None])
        
        df[['area_name', 'city']] = df['location'].apply(split_location_regex)
        
        # Extract seller member since year
        def extract_year(member_str):
            if pd.isna(member_str):
                return np.nan
            match = re.search(r'\b(\d{4})\b', str(member_str))
            return int(match.group(1)) if match else np.nan
        
        df['seller_member_since_year'] = df['seller_member_since'].apply(extract_year)
        
        # Fill missing values
        df['down_payment'] = df['down_payment'].fillna(0)
        df['area'] = df.groupby('property_type')['area'].transform(lambda x: x.fillna(x.mean()))
        df['furnished'] = df['furnished'].fillna(df['furnished'].mode()[0])
        
        def fill_payment_option(row):
            if pd.isna(row['payment_option']):
                if row['down_payment'] == 0:
                    return 'Cash'
                elif row['down_payment'] > 0:
                    return 'Installment'
                else:
                    return 'Cash or Installment'
            return row['payment_option']
        
        df['payment_option'] = df.apply(fill_payment_option, axis=1)
        
        # Drop unnecessary columns
        df.drop(columns=['level'], inplace=True, errors='ignore')
        
        # Handle amenities
        df['amenities'] = df['amenities'].fillna('Unknown')
        df['amenities_count_full'] = df['amenities'].apply(
            lambda x: len([a.strip() for a in x.split(',') if a.strip() != '' and a.strip().lower() != 'unknown'])
        )
        df['has_garden'] = df['amenities'].str.contains('Garden', na=False)
        df['has_security'] = df['amenities'].str.contains('Security', na=False)
        df['has_pool'] = df['amenities'].str.contains('Pool', na=False)
        df['has_balcony'] = df['amenities'].str.contains('Balcony', na=False)
        df['has_parking'] = df['amenities'].str.contains('Covered Parking', na=False)
        df['has_elevator'] = df['amenities'].str.contains('Elevator', na=False)
        
        # Calculate price per sqm
        df['price_per_sqm'] = df['price'] / df['area']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Cache model training
@st.cache_data(show_spinner=False)
def train_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, LabelEncoder]:
    try:
        model_df = df[['city', 'area', 'bedrooms', 'bathrooms', 'price']].dropna()
        le = LabelEncoder()
        model_df['city'] = le.fit_transform(model_df['city'])
        X = model_df[['city', 'area', 'bedrooms', 'bathrooms']]
        y = model_df['price']
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model, le
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

# Cache metrics computation
@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        return {
            'avg_price': df['price'].mean() if not df['price'].isna().all() else 0,
            'total_listings': len(df),
            'avg_price_sqm': df['price_per_sqm'].mean() if not df['price_per_sqm'].isna().all() else 0,
            'avg_area': df['area'].mean() if not df['area'].isna().all() else 0,
            'most_common_type': df['property_type'].mode()[0] if not df.empty else "N/A"
        }
    except Exception as e:
        st.error(f"Error computing metrics: {str(e)}")
        return {}

# Cache data filtering
@st.cache_data(show_spinner=False)
def filter_data(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    try:
        filtered_df = df.copy()
        if filters['search_term']:
            mask = (
                filtered_df['area_name'].str.contains(filters['search_term'], case=False, na=False) | 
                filtered_df['title'].str.contains(filters['search_term'], case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        if filters['city'] != 'All':
            filtered_df = filtered_df[filtered_df['city'] == filters['city']]
        if filters['property_type'] != 'All':
            filtered_df = filtered_df[filtered_df['property_type'] == filters['property_type']]
        filtered_df = filtered_df[
            (filtered_df['price'] >= filters['price_range'][0]) & 
            (filtered_df['price'] <= filters['price_range'][1])
        ]
        if filters['bedrooms']:
            filtered_df = filtered_df[filtered_df['bedrooms'].isin(filters['bedrooms'])]
        if filters['bathrooms']:
            filtered_df = filtered_df[filtered_df['bathrooms'].isin(filters['bathrooms'])]
        if filters['furnished'] != 'All':
            filtered_df = filtered_df[filtered_df['furnished'] == filters['furnished']]
        if filters['payment_option'] != 'All':
            filtered_df = filtered_df[filtered_df['payment_option'] == filters['payment_option']]
        if filters['amenities']:
            amenity_masks = [filtered_df[f'has_{amenity.lower()}'] == True for amenity in filters['amenities']]
            if amenity_masks:
                filtered_df = filtered_df[np.all(amenity_masks, axis=0)]
        if len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            filtered_df = filtered_df[
                (filtered_df['creation_date'].dt.date >= start_date) & 
                (filtered_df['creation_date'].dt.date <= end_date)
            ]
        return filtered_df
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return pd.DataFrame()

# Cache visualizations
@st.cache_data(show_spinner=False)
def create_price_distribution_plot(df: pd.DataFrame) -> go.Figure:
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['price'], nbinsx=30, name='Price', marker_color='#2c3e50'))
        fig.update_layout(title="Price Distribution", xaxis_title="Price (EGP)", yaxis_title="Count",
                         margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price distribution plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_boxplot(df: pd.DataFrame) -> go.Figure:
    try:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['price'], name='Price', marker_color='#3498db'))
        fig.update_layout(title="Price Distribution (Boxplot)", yaxis_title="Price (EGP)",
                         margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price boxplot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_property_type_plot(df: pd.DataFrame) -> go.Figure:
    try:
        fig = px.box(df, x='property_type', y='price', title="Price by Property Type",
                    labels={'price': 'Price (EGP)', 'property_type': 'Property Type'},
                    color_discrete_sequence=['#3498db'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating property type plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_feature_importance_plot(df: pd.DataFrame) -> go.Figure:
    try:
        le = LabelEncoder()
        df['location_encoded'] = le.fit_transform(df['location'].fillna('Unknown'))
        features = ['has_garden', 'has_security', 'has_pool', 'has_balcony', 'has_parking', 'has_elevator', 
                   'area', 'bedrooms', 'bathrooms', 'location_encoded']
        model_df = df[features + ['price']].dropna()
        X = model_df[features]
        y = model_df['price']
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig = px.bar(importance_df, x='Importance', y='Feature', title="Feature Importance for Price Prediction",
                    color_discrete_sequence=['#e74c3c'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating feature importance plot: {str(e)}")
        return go.Figure()

# Display metrics
def display_metrics(metrics: Dict[str, Any], overall_df: pd.DataFrame) -> None:
    try:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_price = metrics.get('avg_price', 0)
            overall_avg = overall_df['price'].mean()
            delta = ((avg_price - overall_avg) / overall_avg * 100) if overall_avg else 0
            st.metric(label="💸 Average Price", 
                     value=f"{avg_price:,.0f} EGP" if avg_price else "N/A",
                     delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            total_listings = metrics.get('total_listings', 0)
            overall_listings = len(overall_df)
            delta = ((total_listings - overall_listings) / overall_listings * 100) if overall_listings else 0
            st.metric(label="🏠 Total Listings", 
                     value=total_listings,
                     delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_price_sqm = metrics.get('avg_price_sqm', 0)
            overall_avg_sqm = overall_df['price_per_sqm'].mean()
            delta = ((avg_price_sqm - overall_avg_sqm) / overall_avg_sqm * 100) if overall_avg_sqm else 0
            st.metric(label="📏 Price per m²", 
                     value=f"{avg_price_sqm:,.0f} EGP" if avg_price_sqm else "N/A",
                     delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_area = metrics.get('avg_area', 0)
            overall_avg_area = overall_df['area'].mean()
            delta = ((avg_area - overall_avg_area) / overall_avg_area * 100) if overall_avg_area else 0
            st.metric(label="🏠 Average Area", 
                     value=f"{avg_area:,.0f} m²" if avg_area else "N/A",
                     delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

# Main app logic
def main():
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_data()
            if not df.empty:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.metrics = compute_metrics(df)
    else:
        df = st.session_state.df

    if df.empty:
        st.error("No data available. Please check the data source.")
        return

    # Title and Introduction
    st.title("🏡 Dubizzle Egypt Property Market Analysis")
    st.markdown("""
        This interactive dashboard provides comprehensive insights into the Egyptian property market based on data from Dubizzle.
        Explore trends, patterns, and key metrics for apartments across various cities. Use the filters to customize your analysis! 📊
    """)

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters & Controls")
        st.markdown("---")
        search_term = st.text_input("🔎 Search by Area or Title", placeholder="e.g., Madinaty")
        city = st.selectbox("🏙️ Select City", options=['All'] + sorted(df['city'].dropna().unique()))
        property_type = st.selectbox("🏠 Property Type", options=['All'] + sorted(df['property_type'].dropna().unique()))
        price_range = st.slider("💰 Price Range (EGP)", 
                              int(df['price'].min()), 
                              int(df['price'].max()), 
                              (int(df['price'].min()), int(df['price'].max())))
        bedrooms = st.multiselect("🛏️ Number of Bedrooms", options=sorted(df['bedrooms'].dropna().unique()))
        bathrooms = st.multiselect("🛁 Number of Bathrooms", options=sorted(df['bathrooms'].dropna().unique()))
        furnished = st.selectbox("🛋️ Furnished Status", options=['All', 'Furnished', 'Unfurnished'])
        payment_option = st.selectbox("💳 Payment Option", options=['All', 'Cash', 'Installment', 'Cash or Installment'])
        amenities = st.multiselect("🏊 Amenities", 
                                 options=['Garden', 'Security', 'Pool', 'Balcony', 'Parking', 'Elevator'])
        min_date = df['creation_date'].min().date() if not df['creation_date'].isna().all() else datetime.now().date()
        max_date = df['creation_date'].max().date() if not df['creation_date'].isna().all() else datetime.now().date()
        date_range = st.date_input("📅 Listing Date Range", 
                                 [min_date, max_date], 
                                 min_value=min_date, 
                                 max_value=max_date)

    # Apply filters
    filters = {
        'search_term': search_term,
        'city': city,
        'property_type': property_type,
        'price_range': price_range,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'furnished': furnished,
        'payment_option': payment_option,
        'amenities': amenities,
        'date_range': date_range
    }
    
    with st.spinner("Applying filters..."):
        filtered_df = filter_data(df, filters)
        st.session_state.filtered_data = filtered_df

    # Key Metrics
    st.markdown("### 📈 Key Market Metrics")
    if st.session_state.metrics:
        display_metrics(compute_metrics(filtered_df), df)
    else:
        st.warning("No metrics available for the selected filters.")

    # Price Predictor
    with st.expander("🔮 Predict Property Price", expanded=False):
        st.subheader("Estimate Property Price")
        city_pred = st.selectbox("Select City for Prediction", sorted(df['city'].dropna().unique()))
        area_pred = st.number_input("Area (m²)", min_value=0, value=100)
        bedrooms_pred = st.number_input("Number of Bedrooms", min_value=0, value=2)
        bathrooms_pred = st.number_input("Number of Bathrooms", min_value=0, value=1)
        
        if st.button("Predict Price"):
            with st.spinner("Calculating..."):
                model, le = train_model(df)
                if model is not None and le is not None:
                    input_data = pd.DataFrame({
                        'city': [le.transform([city_pred])[0]],
                        'area': [area_pred],
                        'bedrooms': [bedrooms_pred],
                        'bathrooms': [bathrooms_pred]
                    })
                    predicted_price = model.predict(input_data)[0]
                    st.success(f"Estimated Price: **{predicted_price:,.0f} EGP**")
                else:
                    st.error("Unable to make prediction due to model training error")

    # Market Overview Section
    st.markdown("---")
    st.header("📊 Market Overview")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Price Analysis", "📍 Location Analysis", 
                                           "🏠 Property Features", "📈 Market Trends", "🔍 Additional Insights"])

    with tab1:
        st.subheader("Price Analysis")
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Try adjusting the filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_price_distribution_plot(filtered_df)
                st.plotly_chart(fig1, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig1.to_image(format="png"), 
                                 file_name="price_distribution.png", 
                                 mime="image/png")
            
            with col2:
                fig2 = create_price_boxplot(filtered_df)
                st.plotly_chart(fig2, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig2.to_image(format="png"), 
                                 file_name="price_boxplot.png", 
                                 mime="image/png")

            col1, col2 = st.columns(2)
            with col1:
                fig3 = create_property_type_plot(filtered_df)
                st.plotly_chart(fig3, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig3.to_image(format="png"), 
                                 file_name="price_by_type.png", 
                                 mime="image/png")
            
            with col2:
                fig4 = px.scatter(filtered_df, x='down_payment', y='price', title="Price vs Down Payment",
                                 labels={'down_payment': 'Down Payment (EGP)', 'price': 'Price (EGP)'},
                                 color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig4, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig4.to_image(format="png"), 
                                 file_name="price_vs_down_payment.png", 
                                 mime="image/png")

            avg_down_payment = filtered_df.groupby('property_type')['down_payment'].mean().reset_index()
            fig5 = px.bar(avg_down_payment, x='property_type', y='down_payment', 
                         title="Average Down Payment by Property Type",
                         labels={'property_type': 'Property Type', 'down_payment': 'Average Down Payment (EGP)'},
                         color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig5, use_container_width=True)
            st.download_button("📥 Download Chart", 
                             data=fig5.to_image(format="png"), 
                             file_name="avg_down_payment.png", 
                             mime="image/png")

    with tab2:
        st.subheader("Location Analysis")
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Try adjusting the filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                top_areas = filtered_df['area_name'].value_counts().head(10)
                fig6 = px.bar(x=top_areas.values, y=top_areas.index, 
                             title="Top 10 Areas by Number of Listings",
                             labels={'x': 'Number of Listings', 'y': 'Area Name'},
                             color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig6, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig6.to_image(format="png"), 
                                 file_name="top_areas.png", 
                                 mime="image/png")
            
            with col2:
                avg_price_city = filtered_df.groupby('city')['price'].mean().sort_values(ascending=False).head(10)
                fig7 = px.bar(x=avg_price_city.values, y=avg_price_city.index, 
                             title="Top 10 Cities by Average Price",
                             labels={'x': 'Average Price (EGP)', 'y': 'City'},
                             color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig7, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig7.to_image(format="png"), 
                                 file_name="avg_price_city.png", 
                                 mime="image/png")

            col1, col2 = st.columns(2)
            with col1:
                top_expensive = filtered_df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
                fig8 = px.bar(x=top_expensive.values, y=top_expensive.index, 
                             title="Top 10 Most Expensive Areas",
                             labels={'x': 'Average Price (EGP)', 'y': 'Area'},
                             color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig8, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig8.to_image(format="png"), 
                                 file_name="top_expensive_areas.png", 
                                 mime="image/png")
            
            with col2:
                top_cheap = filtered_df.groupby('location')['price'].mean().sort_values().head(10)
                fig9 = px.bar(x=top_cheap.values, y=top_cheap.index, 
                             title="Top 10 Cheapest Areas",
                             labels={'x': 'Average Price (EGP)', 'y': 'Area'},
                             color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig9, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                 data=fig9.to_image(format="png"), 
                                 file_name="top_cheap_areas.png", 
                                 mime="image/png")

    with tab3:
        st.subheader("Property Features Analysis")
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Try adjusting the filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig10 = px.box(filtered_df, x='bedrooms', y='price',
                              title="Price Distribution by Number of Bedrooms",
                              labels={'price': 'Price (EGP)', 'bedrooms': 'Number of Bedrooms'},
                              color_discrete_sequence=['#9b59b6'])
                st.plotly_chart(fig10, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig10.to_image(format="png"), 
                                  file_name="price_by_bedrooms.png", 
                                  mime="image/png")
            
            with col2:
                fig11 = px.box(filtered_df, x='amenities_count_full', y='price',
                              title="Price by Number of Amenities",
                              labels={'amenities_count_full': 'Number of Amenities', 'price': 'Price (EGP)'},
                              color_discrete_sequence=['#f1c40f'])
                st.plotly_chart(fig11, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig11.to_image(format="png"), 
                                  file_name="price_by_amenities.png", 
                                  mime="image/png")

            scatter_df = filtered_df.copy()
            scatter_df['bedrooms'] = scatter_df['bedrooms'].fillna(0)
            fig12 = px.scatter(scatter_df, x='area', y='price', color='property_type', size='bedrooms',
                              title="Price vs Area (by Property Type)",
                              labels={'area': 'Area (m²)', 'price': 'Price (EGP)'},
                              color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig12, use_container_width=True)
            st.download_button("📥 Download Chart", 
                              data=fig12.to_image(format="png"), 
                              file_name="price_vs_area.png", 
                              mime="image/png")

            col1, col2 = st.columns(2)
            with col1:
                fig13 = px.scatter(scatter_df, x='area', y='bedrooms', color='property_type',
                                  title="Bedrooms vs Area by Property Type",
                                  labels={'area': 'Area (m²)', 'bedrooms': 'Number of Bedrooms'},
                                  color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig13, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig13.to_image(format="png"), 
                                  file_name="bedrooms_vs_area.png", 
                                  mime="image/png")
            
            with col2:
                fig14 = px.box(filtered_df, x='ownership', y='price',
                              title="Price by Ownership Type",
                              labels={'ownership': 'Ownership Type', 'price': 'Price (EGP)'},
                              color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig14, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig14.to_image(format="png"), 
                                  file_name="price_by_ownership.png", 
                                  mime="image/png")

    with tab4:
        st.subheader("Market Trends")
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Try adjusting the filters.")
        else:
            daily_counts = filtered_df['creation_date'].value_counts().sort_index()
            fig15 = px.line(x=daily_counts.index, y=daily_counts.values,
                           title="Number of Listings per Day",
                           labels={'x': 'Date', 'y': 'Number of Listings'},
                           color_discrete_sequence=['#2c3e50'])
            fig15.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig15, use_container_width=True)
            st.download_button("📥 Download Chart", 
                              data=fig15.to_image(format="png"), 
                              file_name="listings_per_day.png", 
                              mime="image/png")

            daily_prices = filtered_df.groupby(filtered_df['creation_date'].dt.date)['price'].mean()
            fig16 = px.line(x=daily_prices.index, y=daily_prices.values,
                           title="Average Price Trend Over Time",
                           labels={'x': 'Date', 'y': 'Average Price (EGP)'},
                           color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig16, use_container_width=True)
            st.download_button("📥 Download Chart", 
                              data=fig16.to_image(format="png"), 
                              file_name="price_trend.png", 
                              mime="image/png")

            fig17, ax = plt.subplots()
            corr = filtered_df[['price', 'area', 'bedrooms', 'bathrooms', 'amenities_count_full']].corr()
            sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
            st.pyplot(fig17)
            buffer = BytesIO()
            fig17.savefig(buffer, format="png")
            st.download_button("📥 Download Chart", 
                              data=buffer.getvalue(), 
                              file_name="correlation_heatmap.png", 
                              mime="image/png")

    with tab5:
        st.subheader("Additional Insights")
        if filtered_df.empty:
            st.warning("No data matches the selected filters. Try adjusting the filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ownership_counts = filtered_df['ownership'].value_counts()
                fig18 = px.pie(values=ownership_counts.values, names=ownership_counts.index,
                              title="Ownership Type Distribution",
                              color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig18, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig18.to_image(format="png"), 
                                  file_name="ownership_distribution.png", 
                                  mime="image/png")
            
            with col2:
                property_counts = filtered_df['property_type'].value_counts()
                fig19 = px.bar(x=property_counts.values, y=property_counts.index,
                              title="Property Type Distribution",
                              labels={'x': 'Count', 'y': 'Property Type'},
                              color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig19, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig19.to_image(format="png"), 
                                  file_name="property_type_distribution.png", 
                                  mime="image/png")

            col1, col2 = st.columns(2)
            with col1:
                furnished_counts = filtered_df['furnished'].value_counts()
                fig20 = px.pie(values=furnished_counts.values, names=furnished_counts.index,
                              title="Furnished vs Unfurnished Distribution",
                              color_discrete_sequence=['#3498db', '#e74c3c'])
                st.plotly_chart(fig20, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig20.to_image(format="png"), 
                                  file_name="furnished_distribution.png", 
                                  mime="image/png")
            
            with col2:
                completion_status_counts = filtered_df['completion_status'].value_counts()
                fig21 = px.bar(x=completion_status_counts.index, y=completion_status_counts.values,
                              title="Completion Status Distribution",
                              labels={'x': 'Completion Status', 'y': 'Count'},
                              color_discrete_sequence=['#f1c40f'])
                st.plotly_chart(fig21, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig21.to_image(format="png"), 
                                  file_name="completion_status_distribution.png", 
                                  mime="image/png")

            completion_status_property_type = pd.crosstab(filtered_df['property_type'], filtered_df['completion_status'])
            fig22 = go.Figure()
            for status in completion_status_property_type.columns:
                fig22.add_trace(go.Bar(
                    y=completion_status_property_type.index,
                    x=completion_status_property_type[status],
                    name=status,
                    orientation='h'
                ))
            fig22.update_layout(title="Completion Status by Property Type",
                               xaxis_title="Count", yaxis_title="Property Type",
                               barmode='stack', margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig22, use_container_width=True)
            st.download_button("📥 Download Chart", 
                              data=fig22.to_image(format="png"), 
                              file_name="completion_status_by_type.png", 
                              mime="image/png")

            col1, col2 = st.columns(2)
            with col1:
                fig23 = px.box(filtered_df, x='completion_status', y='price',
                              title="Price by Completion Status",
                              labels={'completion_status': 'Completion Status', 'price': 'Price (EGP)'},
                              color_discrete_sequence=['#9b59b6'])
                st.plotly_chart(fig23, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig23.to_image(format="png"), 
                                  file_name="price_by_completion_status.png", 
                                  mime="image/png")
            
            with col2:
                fig24 = px.box(filtered_df, x='payment_option', y='price',
                              title="Price by Payment Option",
                              labels={'payment_option': 'Payment Option', 'price': 'Price (EGP)'},
                              color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig24, use_container_width=True)
                st.download_button("📥 Download Chart", 
                                  data=fig24.to_image(format="png"), 
                                  file_name="price_by_payment_option.png", 
                                  mime="image/png")

            fig25 = px.scatter(filtered_df, x='area', y='price', color='payment_option',
                              title="Price vs Area (by Payment Option)",
                              labels={'area': 'Area (m²)', 'price': 'Price (EGP)'},
                              color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig25, use_container_width=True)
            st.download_button("📥 Download Chart", 
                              data=fig25.to_image(format="png"), 
                              file_name="price_vs_area_payment.png", 
                              mime="image/png")

            col1, col2 = st.columns(2)
            with col1:
                room_type_relation = filtered_df.groupby(['bedrooms', 'property_type']).size().unstack(fill_value=0)
                fig26, ax = plt.subplots()
                sns.heatmap(room_type_relation, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                ax.set_title("Bedrooms vs Property Type")
                ax.set_xlabel("Property Type")
                ax.set_ylabel("Number of Bedrooms")
                st.pyplot(fig26)
                buffer = BytesIO()
                fig26.savefig(buffer, format="png")
                st.download_button("📥 Download Chart", 
                                  data=buffer.getvalue(), 
                                  file_name="bedrooms_vs_property_type.png", 
                                  mime="image/png")
            
            with col2:
                bathroom_type_relation = filtered_df.groupby(['bathrooms', 'property_type']).size().unstack(fill_value=0)
                fig27, ax = plt.subplots()
                sns.heatmap(bathroom_type_relation, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Bathrooms vs Property Type")
                ax.set_xlabel("Property Type")
                ax.set_ylabel("Number of Bathrooms")
                st.pyplot(fig27)
                buffer = BytesIO()
                fig27.savefig(buffer, format="png")
                st.download_button("📥 Download Chart", 
                                  data=buffer.getvalue(), 
                                  file_name="bathrooms_vs_property_type.png", 
                                  mime="image/png")

            fig28 = create_feature_importance_plot(filtered_df)
            st.plotly_chart(fig28, use_container_width=True)
            st.download_button("📥 Download Chart", 
                              data=fig28.to_image(format="png"), 
                              file_name="feature_importance.png", 
                              mime="image/png")

    # Data Table and Download
    st.markdown("---")
    st.header("📋 Filtered Listings")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.dataframe(filtered_df, use_container_width=True, height=300)
        def to_csv(df):
            return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
        csv = to_csv(filtered_df)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="filtered_apartments.csv",
            mime="text/csv"
        )

    # Market Insights and Fun Facts
    st.markdown("---")
    st.header("💡 Market Insights & Fun Facts")
    if not filtered_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
            most_expensive_area = filtered_df.groupby('area_name')['price'].mean().idxmax()
            most_expensive_price = filtered_df.groupby('area_name')['price'].mean().max()
            st.markdown(f"**Most Expensive Area** 🏰<br>{most_expensive_area}: {most_expensive_price:,.0f} EGP")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
            cheapest_area = filtered_df.groupby('area_name')['price'].mean().idxmin()
            cheapest_price = filtered_df.groupby('area_name')['price'].mean().min()
            st.markdown(f"**Cheapest Area** 💸<br>{cheapest_area}: {cheapest_price:,.0f} EGP")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
            common_amenity = filtered_df[['has_garden', 'has_security', 'has_pool', 
                                       'has_balcony', 'has_parking', 'has_elevator']].sum().idxmax()
            common_amenity = common_amenity.replace('has_', '').capitalize()
            st.markdown(f"**Most Common Amenity** 🏊<br>{common_amenity}")
            st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Key Findings")
            avg_bedrooms = filtered_df['bedrooms'].mean() if not filtered_df['bedrooms'].isna().all() else 0
            st.markdown(f"""
            - **Most Expensive Area**: {most_expensive_area}
            - **Average Price per m²**: {filtered_df['price_per_sqm'].mean():,.0f} EGP
            - **Most Common Property Type**: {filtered_df['property_type'].mode()[0]}
            - **Average Number of Bedrooms**: {avg_bedrooms:.1f}
            """)
        
        with col2:
            st.markdown("### 📈 Market Trends")
            price_trend = 'Increasing' if filtered_df['price'].mean() > df['price'].mean() else 'Decreasing'
            listings_growth = ((len(filtered_df) - len(df)) / len(df) * 100)
            days_on_market = (filtered_df['creation_date'].max() - filtered_df['creation_date'].min()).days
            st.markdown(f"""
            - **Price Trend**: {price_trend}
            - **Listings Growth**: {listings_growth:.1f}%
            - **Average Days on Market**: {days_on_market} days
            """)
    else:
        st.markdown("Apply filters to discover market insights!")

if __name__ == "__main__":
    main()
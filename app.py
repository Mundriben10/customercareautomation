import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# import pickle
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
# import re

# Configuration
MODEL_PATH = "banking_model.pkl"
DATA_PATH = "banking_data.csv"
VECTORIZER_PATH = "vectorizer.pkl"

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = 0
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0

# Sample banking data for initial training
SAMPLE_BANKING_DATA = [
    {"problem": "I forgot my ATM PIN", "category": "ATM Issues", "solution": "Visit nearest branch with ID proof or reset PIN through mobile banking app", "steps": ["Visit branch with valid ID", "Fill PIN reset form", "Verify identity", "Set new PIN"]},
    {"problem": "My account is blocked", "category": "Account Issues", "solution": "Contact customer service immediately to unblock account", "steps": ["Call customer service", "Provide account details", "Verify identity", "Follow unblock procedure"]},
    {"problem": "Transaction failed but money deducted", "category": "Transaction Issues", "solution": "Money will be auto-reversed in 3-5 business days", "steps": ["Check transaction status", "Wait 3-5 business days", "Contact support if not reversed", "Provide transaction reference"]},
    {"problem": "Cannot access mobile banking", "category": "Mobile Banking", "solution": "Reset mobile banking password or update app", "steps": ["Check internet connection", "Update mobile app", "Reset password", "Contact support if needed"]},
    {"problem": "Credit card payment not reflecting", "category": "Credit Card", "solution": "Payment takes 1-2 business days to reflect", "steps": ["Check payment confirmation", "Wait 1-2 business days", "Check credit card statement", "Contact support if not reflected"]},
    {"problem": "Loan EMI auto-debit failed", "category": "Loan Services", "solution": "Ensure sufficient balance for next auto-debit", "steps": ["Check account balance", "Ensure sufficient funds", "Contact loan department", "Make manual payment if needed"]},
    {"problem": "Cheque bounced due to insufficient funds", "category": "Cheque Services", "solution": "Deposit sufficient funds and contact payee", "steps": ["Check account balance", "Deposit required amount", "Pay bounce charges", "Contact payee to re-present cheque"]},
    {"problem": "Net banking locked due to wrong password", "category": "Net Banking", "solution": "Reset password through registered mobile or visit branch", "steps": ["Use forgot password option", "Enter registered mobile number", "Follow OTP verification", "Set new password"]},
    {"problem": "Fixed deposit maturity amount not credited", "category": "Fixed Deposit", "solution": "Check maturity date and contact branch", "steps": ["Verify maturity date", "Check account statement", "Contact home branch", "Provide FD receipt"]},
    {"problem": "Debit card swipe declined", "category": "Debit Card", "solution": "Check card limit and account balance", "steps": ["Check daily transaction limit", "Verify account balance", "Contact customer service", "Request limit increase if needed"]},
    {"problem": "International transaction blocked", "category": "International Services", "solution": "Enable international transactions or inform bank", "steps": ["Enable international usage", "Inform bank about travel", "Check transaction limits", "Contact support for activation"]},
    {"problem": "SMS alerts not received", "category": "SMS Services", "solution": "Update mobile number or reactivate SMS service", "steps": ["Check registered mobile number", "Update mobile number if changed", "Reactivate SMS service", "Contact support"]},
    {"problem": "Interest rate query on savings account", "category": "Savings Account", "solution": "Check current interest rates on bank website", "steps": ["Visit bank website", "Check interest rate section", "Contact relationship manager", "Consider higher interest accounts"]},
    {"problem": "Want to close bank account", "category": "Account Closure", "solution": "Visit branch with closure form and clear all dues", "steps": ["Download closure form", "Clear all pending dues", "Visit branch with documents", "Submit closure request"]},
    {"problem": "Pension not credited to account", "category": "Pension Services", "solution": "Check pension schedule and contact pension department", "steps": ["Verify pension schedule", "Check account statement", "Contact pension department", "Submit life certificate if required"]}
]

class IncrementalBankingModel:
    def _init_(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
    def train_model(self, data):
        """Train the model with new data"""
        try:
            df = pd.DataFrame(data)
            X = df['problem'].values
            y = df['category'].values
            
            # Create pipeline with TF-IDF and Naive Bayes
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.model = MultinomialNB()
            
            # Transform text data
            X_transformed = self.vectorizer.fit_transform(X)
            
            # Train model
            self.model.fit(X_transformed, y)
            self.is_trained = True
            
            # Calculate accuracy
            y_pred = self.model.predict(X_transformed)
            accuracy = accuracy_score(y, y_pred)
            
            return accuracy
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return 0
    
    def incremental_train(self, new_data):
        """Add new data and retrain model"""
        if not self.is_trained:
            return self.train_model(new_data)
        
        try:
            df = pd.DataFrame(new_data)
            X_new = df['problem'].values
            y_new = df['category'].values
            
            # Transform new data
            X_new_transformed = self.vectorizer.transform(X_new)
            
            # Partial fit for incremental learning
            self.model.partial_fit(X_new_transformed, y_new)
            
            return True
        except Exception as e:
            st.error(f"Incremental training error: {str(e)}")
            return False
    
    def predict_category(self, problem_text):
        """Predict category for given problem"""
        if not self.is_trained:
            return None, 0
        
        try:
            problem_transformed = self.vectorizer.transform([problem_text])
            prediction = self.model.predict(problem_transformed)[0]
            probability = self.model.predict_proba(problem_transformed).max()
            
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, 0
    
    def get_solution(self, problem_text, training_data):
        """Get solution for predicted category"""
        category, confidence = self.predict_category(problem_text)
        
        if category:
            # Find solution from training data
            for item in training_data:
                if item['category'] == category:
                    return {
                        'category': category,
                        'confidence': confidence,
                        'solution': item['solution'],
                        'steps': item['steps']
                    }
        
        return None

# Initialize model
@st.cache_resource
def load_model():
    return IncrementalBankingModel()

model = load_model()

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .solution-steps {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .animated-title {
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown('<div class="main-header"><h1 class="animated-title">🏦 Banking Customer Care AI Assistant</h1><p>Intelligent Banking Support with Continuous Learning</p></div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Choose a section", ["🤖 Customer Query", "📚 Train Model", "📊 Analytics", "⚙️ Settings"])

if page == "🤖 Customer Query":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Customer Support Chat")
    
    # Initialize with sample data if not trained
    if not st.session_state.model_trained:
        with st.spinner("🔄 Initializing AI model with banking data..."):
            accuracy = model.train_model(SAMPLE_BANKING_DATA)
            st.session_state.model_trained = True
            st.session_state.training_data = SAMPLE_BANKING_DATA
            st.session_state.model_accuracy = accuracy
            time.sleep(2)
        st.success("✅ AI model ready for banking support!")
    
    # Customer query input
    customer_query = st.text_area("🔍 Describe your banking issue:", 
                                  placeholder="e.g., I forgot my ATM PIN and need to reset it",
                                  height=100)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Get Solution"):
            if customer_query:
                with st.spinner("🔍 Analyzing your issue..."):
                    time.sleep(1)
                    solution = model.get_solution(customer_query, st.session_state.training_data)
                    st.session_state.total_queries += 1
                
                if solution:
                    st.markdown('<div class="success-card">', unsafe_allow_html=True)
                    st.markdown(f"*🎯 Category:* {solution['category']}")
                    st.markdown(f"*🎯 Confidence:* {solution['confidence']:.2%}")
                    st.markdown(f"*💡 Solution:* {solution['solution']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="solution-steps">', unsafe_allow_html=True)
                    st.markdown("*📋 Steps to Resolution:*")
                    for i, step in enumerate(solution['steps'], 1):
                        st.markdown(f"{i}. {step}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Unable to categorize the issue. Please provide more details or train the model with relevant data.")
            else:
                st.warning("⚠️ Please enter your banking issue to get assistance.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📚 Train Model":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎓 Continuous Learning System")
    
    # Display current model status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Status", "✅ Active" if st.session_state.model_trained else "❌ Not Trained")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Records", len(st.session_state.training_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add new training data
    st.markdown("### ➕ Add New Training Data")
    
    with st.form("training_form"):
        new_problem = st.text_input("🔍 Customer Problem:", placeholder="e.g., Cannot transfer money online")
        new_category = st.selectbox("🏷️ Category:", 
                                   ["ATM Issues", "Account Issues", "Transaction Issues", "Mobile Banking", 
                                    "Credit Card", "Loan Services", "Cheque Services", "Net Banking", 
                                    "Fixed Deposit", "Debit Card", "International Services", "SMS Services", 
                                    "Savings Account", "Account Closure", "Pension Services", "Other"])
        new_solution = st.text_area("💡 Solution:", placeholder="Provide step-by-step solution")
        new_steps = st.text_area("📋 Resolution Steps:", placeholder="Enter steps separated by commas")
        
        submitted = st.form_submit_button("🚀 Add Training Data")
        
        if submitted:
            if new_problem and new_category and new_solution:
                steps_list = [step.strip() for step in new_steps.split(',') if step.strip()]
                new_data = {
                    "problem": new_problem,
                    "category": new_category,
                    "solution": new_solution,
                    "steps": steps_list
                }
                
                st.session_state.training_data.append(new_data)
                
                # Incremental training
                with st.spinner("🔄 Training model with new data..."):
                    if st.session_state.model_trained:
                        success = model.incremental_train([new_data])
                        if success:
                            st.success("✅ Model updated with new data!")
                        else:
                            st.error("❌ Failed to update model")
                    else:
                        accuracy = model.train_model(st.session_state.training_data)
                        st.session_state.model_trained = True
                        st.session_state.model_accuracy = accuracy
                        st.success("✅ Model trained successfully!")
            else:
                st.error("❌ Please fill all required fields")
    
    # Bulk training option
    st.markdown("### 📁 Bulk Training Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file with training data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("🔄 Train with Uploaded Data"):
                with st.spinner("🔄 Training model with uploaded data..."):
                    training_data = df.to_dict('records')
                    accuracy = model.train_model(training_data)
                    st.session_state.training_data.extend(training_data)
                    st.session_state.model_trained = True
                    st.session_state.model_accuracy = accuracy
                    st.success(f"✅ Model trained with {len(training_data)} records! Accuracy: {accuracy:.2%}")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📊 Analytics":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 System Analytics")
    
    if st.session_state.training_data:
        df = pd.DataFrame(st.session_state.training_data)
        
        # Category distribution
        category_counts = df['category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(values=category_counts.values, 
                            names=category_counts.index, 
                            title="📊 Issue Categories Distribution")
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(x=category_counts.index, 
                            y=category_counts.values,
                            title="📈 Issues by Category")
            fig_bar.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Performance metrics
        st.markdown("### 🎯 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Categories", len(category_counts))
        with col2:
            st.metric("Total Queries Processed", st.session_state.total_queries)
        with col3:
            st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
        with col4:
            st.metric("Training Records", len(st.session_state.training_data))
        
        # Recent training data
        st.markdown("### 📋 Recent Training Data")
        recent_data = df.tail(10)
        st.dataframe(recent_data, use_container_width=True)
    
    else:
        st.info("📊 No training data available. Please add some training data to see analytics.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "⚙️ Settings":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ System Settings")
    
    # Model configuration
    st.markdown("#### 🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.7)
        max_features = st.number_input("📊 Max TF-IDF Features", 1000, 10000, 5000)
    
    with col2:
        auto_retrain = st.checkbox("🔄 Auto-retrain on new data", True)
        save_model = st.checkbox("💾 Save model locally", True)
    
    # Export/Import settings
    st.markdown("#### 📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Training Data"):
            if st.session_state.training_data:
                df = pd.DataFrame(st.session_state.training_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"banking_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No training data to export")
    
    with col2:
        if st.button("🗑️ Reset Model"):
            st.session_state.model_trained = False
            st.session_state.training_data = []
            st.session_state.model_accuracy = 0
            st.session_state.total_queries = 0
            st.success("✅ Model reset successfully!")
    
    # System information
    st.markdown("#### ℹ️ System Information")
    
    info_data = {
        "Feature": ["Model Type", "Vectorizer", "Training Records", "Categories", "Last Updated"],
        "Value": [
            "Multinomial Naive Bayes",
            "TF-IDF Vectorizer",
            len(st.session_state.training_data),
            len(set(item['category'] for item in st.session_state.training_data)) if st.session_state.training_data else 0,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    }
    
    st.table(pd.DataFrame(info_data))
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        🏦 Banking Customer Care AI Assistant | Built with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
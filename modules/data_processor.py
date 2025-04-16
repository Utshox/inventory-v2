import pandas as pd
import numpy as np
import os
import re

class DataProcessor:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.df = None
        self.current_file = None
        self.error = None

    def load_data(self, filename):
        try:
            file_path = os.path.join(self.upload_folder, filename)
            self.df = pd.read_csv(file_path)
            
            # Clean and prepare the data
            self._clean_data()
            
            self.current_file = filename
            return self.df
        except Exception as e:
            self.error = f"CSV Error: {str(e)}"
            raise

    def _clean_data(self):
        """Clean and prepare the DataFrame"""
        if self.df is None or self.df.empty:
            return
            
        # Standardize column names
        self.df.columns = [self._clean_column_name(col) for col in self.df.columns]
        
        # Ensure critical columns exist
        required_cols = {'Product ID', 'Product Name', 'Unit Price'}
        existing_cols = set(self.df.columns)
        
        # Attempt to map similar columns to required ones
        column_mapping = {
            'Product ID': ['productid', 'id', 'sku', 'item number', 'product number', 'service id', 'serviceid'],
            'Product Name': ['productname', 'name', 'description', 'item', 'product', 'service name', 'servicename'],
            'Unit Price': ['unitprice', 'price', 'cost', 'retail price', 'msrp', 'service price', 'serviceprice', 'fee']
        }
        
        for required, alternatives in column_mapping.items():
            if required not in existing_cols:
                # Try to find a matching column
                for col in self.df.columns:
                    if any(alt in col.lower().replace(" ", "") for alt in alternatives):
                        self.df = self.df.rename(columns={col: required})
                        break
        
        # Replace empty strings with NaN
        self.df = self.df.replace('', np.nan)
        
        # Handle price formatting
        price_cols = [col for col in self.df.columns if any(term in col.lower() for term in ['price', 'cost', 'total', 'fee'])]
        for col in price_cols:
            if self.df[col].dtype == object:  # If string type
                # First convert string values to numeric
                self.df[col] = self.df[col].apply(lambda x: self._extract_numeric_value(x) if pd.notnull(x) else np.nan)
            
            # Fill any NaN values with zero to prevent calculation errors
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0.0)
    
    def _clean_column_name(self, name):
        """Standardize column name formatting"""
        # Remove special characters, convert to title case
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name).strip().title()
        return clean_name.replace(' ', ' ')  # Ensure consistent spacing
    
    def _extract_numeric_value(self, value):
        """Extract numeric value from string with currency symbols"""
        if not isinstance(value, str):
            return value
            
        # Remove currency symbols and commas
        numeric_str = re.sub(r'[^\d.]', '', value)
        try:
            return float(numeric_str) if numeric_str else 0.0
        except ValueError:
            return 0.0  # Return 0 instead of NaN

    def get_columns(self):
        """Get list of DataFrame columns"""
        return self.df.columns.tolist() if self.df is not None else []
    
    def get_sample_data(self, n=5):
        """Get sample data for display"""
        try:
            if self.df is None or self.df.empty:
                return []
                
            sample = self.df.head(n).copy()
            
            # Format numeric fields
            numeric_cols = sample.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                sample[col] = sample[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "0.00")
                
            return sample.to_dict('records')
            
        except Exception as e:
            print(f"Sample data error: {str(e)}")
            return []
    
    def is_loaded(self):
        """Check if data is loaded"""
        return self.df is not None
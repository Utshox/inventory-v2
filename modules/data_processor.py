import pandas as pd
import os

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
            # Add dummy data if needed for testing
            if self.df.empty:
                self.df = pd.DataFrame({
                    'Product ID': ['PD-1001', 'PD-1002'],
                    'Product Name': ['Test Door', 'Test Frame'],
                    'Unit Price': [100.50, 200.75]
                })
            return self.df
        except Exception as e:
            self.error = f"CSV Error: {str(e)}"
            raise

    def get_columns(self):
        return self.df.columns.tolist() if self.df is not None else []
    
    def get_sample_data(self, n=5):
        """Safe sample data retrieval"""
        try:
            if self.df is None or self.df.empty:
                return []
                
            sample = self.df.head(n).copy()
            
            # Clean numeric fields
            numeric_cols = sample.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                sample[col] = sample[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                
            return sample.to_dict('records')
            
        except Exception as e:
            print(f"Sample data error: {str(e)}")
            return []
    
    def is_loaded(self):
        return self.df is not None
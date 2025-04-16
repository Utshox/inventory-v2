# modules/ai_agent.py
import os
import re
import pandas as pd
import io
from dotenv import load_dotenv
from google.generativeai import configure
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

class AIAgent:
    def __init__(self, df, api_key=None):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not loaded")
            
        # Configure Gemini API with provided key or environment variable
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key is not set")
            
        configure(api_key=api_key)
        
        self.df = df
        self.agent = self.create_agent()
        
    def create_agent(self):
        from langchain.prompts import PromptTemplate
        
        # Define the custom prefix directly in the method
        custom_prefix = f"""
        You are a data analysis assistant working with commercial door product data.
        The dataset contains these columns: {{columns}}.

        **Strict Instructions:**
        1. ALWAYS use the existing DataFrame `df` (already loaded)
        2. Never generate new sample data - use only the provided data
        3. Format results as markdown tables
        4. Never mention tool names or execution methods
        5. For random sampling, use: df.sample(n=4)

        Example Response Format:
        Here are 4 random products from the dataset:

        | Product ID | Product Name       | Unit Price |
        |------------|--------------------|------------|
        | PD-1001    | Steel Security Door| $1,200.00  |
        | PD-1023    | Glass Store Front  | $2,850.00  |
        ... (2 more rows)
        """

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1
        )
        
        return create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=True,
            prefix=custom_prefix,
            input_variables=["columns"],
            include_df_in_prompt=False,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True  # Add this parameter to enable Python REPL execution
        )
    def process_query(self, query):
        """Process natural language query with enhanced handling"""
        try:
            enhanced_query = self._enhance_query(query)
            
            # Append specific instructions to avoid limitation messages
            enhanced_query += """
            Important: DO NOT mention any limitations about executing code. 
            Process the query directly using the data.
            Always return actual results, not placeholder text.
            Do not include phrases like "I am unable to execute the code" or "Due to limitations".
            """
            
            response = self.agent.run({
                "input": enhanced_query,
                "columns": ", ".join(self.df.columns.tolist())
            })
            
            # Clean up the response to remove any remaining limitation statements
            response = re.sub(
                r'(?i)(due to (?:the )?limitations|unable to execute|I cannot directly|I am not able to)', 
                '', 
                response
            )
            
            # Make sure we have results, not just an explanation of what would happen
            if "would generate" in response.lower() or "would display" in response.lower():
                # Force execution by appending concrete instructions
                retry_query = enhanced_query + "\nExecute this query against the actual data and show results."
                response = self.agent.run({
                    "input": retry_query,
                    "columns": ", ".join(self.df.columns.tolist())
                })
            
            return response, self.extract_table(response)
        except Exception as e:
            error_msg = str(e)
            # Don't expose internal errors to the user
            user_friendly_error = "There was an issue processing your query. Please try rephrasing it."
            print(f"Query processing error: {error_msg}")
            return user_friendly_error, None

    def _enhance_query(self, query):
        """Improve query understanding for better results"""
        # Numerical handling
        query = re.sub(
            r'(\d+)\s*(most|top|first|last)\s',
            r'first \1 ', 
            query, 
            flags=re.IGNORECASE
        )
        
        # Category handling
        if 'category' in query.lower():
            query += "\nConsider variations of category names (e.g., 'Hardware' vs 'Door Hardware')"
        
        # Sorting instructions
        if 'expensive' in query.lower() or 'price' in query.lower():
            query += "\nSort results by price in descending order"
        
        return query

    def extract_table(self, response):
        """Extract structured data from AI response with enhanced validation"""
        result_df = None
        
        try:
            if not response:
                return []

            # Preprocess response
            modified_response = re.sub(
                r'(```python\s*)', 
                r'\1import pandas as pd\n',
                response,
                flags=re.IGNORECASE
            )

            # Try extracting from code blocks first
            if "```" in modified_response:
                result_df = self._extract_from_code_blocks(modified_response)

            # Fallback to inline tables
            if result_df is None:
                result_df = self._extract_inline_tables(modified_response)

            # Final fallback to numbered lists
            if result_df is None:
                result_df = self._extract_from_numbered_lists(modified_response)

            # Post-process the dataframe
            if result_df is not None and not result_df.empty:
                result_df = self._clean_dataframe(result_df)
                return result_df.to_dict('records')

            return []

        except Exception as e:
            print(f"Table extraction error: {str(e)}")
            return []

    # Helper methods
    def _extract_from_code_blocks(self, response):
        """Extract tables from markdown/code blocks"""
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        
        for block in code_blocks:
            try:
                # Handle markdown tables
                if any('|' in line for line in block.split('\n')):
                    return pd.read_csv(io.StringIO(block), sep='|', skipinitialspace=True).dropna(axis=1, how='all')
                
                # Handle CSV-like data
                return pd.read_csv(io.StringIO(block))
                
            except Exception as e:
                continue
                
        return None

    def _extract_inline_tables(self, response):
        """Improved markdown table parsing"""
        try:
            # Find all potential tables
            tables = re.findall(
                r'(\|.*\|[\n\r]+)\|?[-: \|]+\|?[\n\r]+((?:\|.*\|[\n\r]?)+)',
                response
            )
            
            best_table = None
            max_rows = 0
            
            for header, body in tables:
                # Process header
                headers = [h.strip() for h in header.split('|') if h.strip()]
                
                # Process body rows
                rows = []
                for line in body.split('\n'):
                    line = line.strip()
                    if line.startswith('|'):
                        cells = [c.strip() for c in line.split('|') if c.strip()]
                        if len(cells) == len(headers):
                            rows.append(cells)
                
                # Validate table
                if len(headers) > 1 and len(rows) > 0:
                    if len(rows) > max_rows:
                        best_table = (headers, rows)
                        max_rows = len(rows)
            
            if best_table:
                headers, rows = best_table
                return pd.DataFrame(rows, columns=headers)
                
        except Exception as e:
            print(f"Table extraction error: {str(e)}")
        
        return None
        
    def _extract_from_numbered_lists(self, response):
        """Extract data from numbered or bulleted lists"""
        try:
            # Find numbered lists with product information
            list_items = re.findall(r'(?:\d+\.|\*)\s*(.*?)(?=(?:\d+\.|\*)|$)', response, re.DOTALL)
            
            if list_items:
                # Determine if items contain product data
                product_items = []
                for item in list_items:
                    # Look for product ID, name, and price patterns
                    if re.search(r'([A-Z0-9]{2,}-\d+|[A-Z]{2,}\d+)', item) and re.search(r'\$\d+', item):
                        product_items.append(item.strip())
                
                if product_items:
                    # Extract data using patterns
                    data = []
                    for item in product_items:
                        product_id = re.search(r'([A-Z0-9]{2,}-\d+|[A-Z]{2,}\d+)', item)
                        product_name = re.search(r'[A-Z][a-z]+\s+[A-Za-z\s]+(?=\$|\(|\-)', item)
                        price = re.search(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', item)
                        
                        if product_id and price:
                            data.append({
                                'Product ID': product_id.group(0),
                                'Product Name': product_name.group(0).strip() if product_name else 'Unknown',
                                'Price': price.group(1).replace(',', '')
                            })
                    
                    if data:
                        return pd.DataFrame(data)
            
            return None
                
        except Exception as e:
            print(f"List extraction error: {str(e)}")
            return None

    def _clean_dataframe(self, df):
        """Enhanced data cleaning"""
        # Clean column names
        df.columns = df.columns.str.strip().str.title()
        
        # Convert numeric columns
        numeric_cols = ['Unit Price', 'Price', 'Cost', 'Total']
        for col in df.columns:
            if any(kw in col.lower() for kw in ['price', 'cost', 'total']):
                df[col] = df[col].replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean warranty information
        if 'Warranty Information' in df.columns:
            df['Warranty Information'] = df['Warranty Information'].str.replace(' years', '').str.replace(' year', '').astype(float)
        
        return df.dropna(how='all')


    def validate_response(self, result_df):
        """Ensure dataframe meets minimum requirements"""
        if result_df is None or result_df.empty:
            return False
        required_columns = {'product', 'name', 'price'}
        return any(col in result_df.columns.str.lower() for col in required_columns)
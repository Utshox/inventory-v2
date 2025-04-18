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
            
            # Add more specific instructions to prevent self-referential or incomplete responses
            enhanced_query += """
            Important instructions:
            1. DO NOT mention your limitations or capabilities in the response.
            2. DO NOT include phrases like "I am unable" or "I cannot".
            3. DO NOT end sentences abruptly with incomplete thoughts.
            4. If you can't produce a result, explain precisely what's missing in the data.
            5. Always provide concrete analysis using the available data.
            6. Format results as markdown tables where appropriate.
            7. NEVER include the words "code", "execute", or "limitations" in your response.
            """
            
            # First attempt
            response = self.agent.run({
                "input": enhanced_query,
                "columns": ", ".join(self.df.columns.tolist())
            })
            
            # Check for problematic patterns that indicate a nonsensical or incomplete response
            problematic_patterns = [
                r'(?i)because I am \w{1,10}\.?$',  # Catches "because I am code" and similar truncations
                r'(?i)I am (unable|not able) to',
                r'(?i)I cannot.*because',
                r'(?i)limitations (of|in) (the|my)',
                r'(?i)as an AI',
                r'(?i)I don\'t have the ability',
                r'(?i)tool (usage|execution)',
                r'(?i)due to (my|the) (limitations|constraints)',
                r'(?i)I am a language model',
                r'(?i)as a language model',
                r'(?i)the execution of'
            ]
            
            needs_retry = False
            for pattern in problematic_patterns:
                if re.search(pattern, response):
                    needs_retry = True
                    break
            
            # Check for truncated responses or nonsensical endings
            if response.endswith(('.', ',', ';', ':', '-')) or len(response.split()) < 5:
                needs_retry = True
            
            # Retry with more explicit instructions if needed
            if needs_retry:
                retry_query = enhanced_query + """
                CRITICAL: Your previous response contained restricted phrases or was incomplete.
                
                Focus ONLY on analyzing the data and providing insights.
                1. Start with a clear statement about what the data shows
                2. Provide organized results in markdown table format
                3. End with a brief conclusion about the findings
                """
                
                response = self.agent.run({
                    "input": retry_query,
                    "columns": ", ".join(self.df.columns.tolist())
                })
            
            # Final cleanup of any remaining problematic phrases
            response = self._clean_response(response)
            
            # If we still have no proper content, provide a generic but useful response
            if not self._has_valid_content(response):
                manufacturer_col = next((col for col in self.df.columns if 'manufacturer' in col.lower()), None)
                price_col = next((col for col in self.df.columns if 'price' in col.lower()), None)
                
                if manufacturer_col and price_col:
                    fallback_msg = f"Here's a summary of the products by price: The data contains {len(self.df)} products across {self.df[manufacturer_col].nunique()} manufacturers, with prices ranging from ${self.df[price_col].min():.2f} to ${self.df[price_col].max():.2f}."
                else:
                    fallback_msg = f"The data contains {len(self.df)} records. Please specify which columns you'd like to analyze."
                
                return fallback_msg, self.extract_table(response)
            
            return response, self.extract_table(response)
        
        except Exception as e:
            error_msg = str(e)
            print(f"Query processing error: {error_msg}")
            # Provide a specific and helpful error message
            return "I couldn't complete this analysis. Please try a different query or check if the data contains the necessary information.", None

    def _clean_response(self, response):
        """Clean up problematic patterns in responses"""
        if not response:
            return ""
        
        # Remove any statements about limitations
        cleaned = re.sub(
            r'(?i)(due to (?:the )?limitations|unable to|I cannot|As an AI|I don\'t have the ability|As a language model)',
            '',
            response
        )
        
        # Remove incomplete thoughts ending with "because I am..."
        cleaned = re.sub(r'(?i)because I am \w+\.?$', '.', cleaned)
        
        # Fix double periods that might be created by removals
        cleaned = re.sub(r'\.\.+', '.', cleaned)
        
        # Fix spacing issues
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        
        return cleaned

    def _has_valid_content(self, response):
        """Check if the response has valid content"""
        # Must have some substantial length
        if not response or len(response) < 20:
            return False
            
        # Should contain numbers if this is a data analysis
        has_numbers = bool(re.search(r'\d', response))
        
        # Should have proper sentences
        proper_sentences = len([s for s in response.split('.') if len(s.strip()) > 10]) > 0
        
        # Should have a table or list-like structure for data
        has_structure = '|' in response or '\n- ' in response or bool(re.search(r'\d+\.', response))
        
        return has_numbers and proper_sentences or has_structure

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
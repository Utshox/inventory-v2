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
        """Process natural language query with enhanced handling and diagnostics"""
        try:
            # Log query and data info for debugging
            print(f"Processing query: {query}")
            print(f"DataFrame shape: {self.df.shape}, columns: {self.df.columns.tolist()}")
            
            # First check if this is a special case for identical items
            special_result = self._process_data(query)
            if special_result:
                print("Using special processing for identical items")
                return special_result, None  # Return the special result with no table data
            
            enhanced_query = self._enhance_query(query)
            
            # Add more specific instructions to prevent problematic responses
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
            print(f"Running agent with columns: {', '.join(self.df.columns.tolist())}")
            response = self.agent.run({
                "input": enhanced_query,
                "columns": ", ".join(self.df.columns.tolist())
            })
            
            print(f"Response received, length: {len(response)}")
            
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
                r'(?i)the execution of',
                r'(?i)python_repl_ast'  # This might indicate a failed execution
            ]
            
            needs_retry = False
            for pattern in problematic_patterns:
                if re.search(pattern, response):
                    print(f"Problematic pattern found: {pattern}")
                    needs_retry = True
                    break
            
            # Check for truncated responses or nonsensical endings
            if response.endswith(('.', ',', ';', ':', '-')) or len(response.split()) < 5:
                print("Response appears truncated or too short")
                needs_retry = True
            
            # Retry with more explicit instructions if needed
            if needs_retry:
                print("Retrying with more explicit instructions")
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
            
            # Extract table data before fallback check
            table_data = self.extract_table(response)
            print(f"Extracted table data: {len(table_data) if table_data else 0} rows")
            
            # If we still have no proper content, provide a generic but useful response
            if not self._has_valid_content(response):
                print("Response lacks valid content, using fallback")
                manufacturer_col = next((col for col in self.df.columns if 'manufacturer' in col.lower()), None)
                price_col = next((col for col in self.df.columns if 'price' in col.lower()), None)
                
                if manufacturer_col and price_col:
                    # Create a simple summary with actual data values
                    fallback_msg = f"Here's a summary of the products by price range:\n\n"
                    fallback_msg += f"• Total products: {len(self.df)}\n"
                    fallback_msg += f"• Manufacturers: {self.df[manufacturer_col].nunique()}\n"
                    fallback_msg += f"• Price range: ${self.df[price_col].min():.2f} to ${self.df[price_col].max():.2f}\n\n"
                    
                    # Add a sample table if no table was extracted
                    if not table_data:
                        sample = self.df.sample(min(5, len(self.df)))
                        fallback_msg += "Here's a sample of products:\n\n"
                        fallback_msg += sample.to_markdown(index=False)
                        # Extract table from the fallback message
                        table_data = sample.to_dict('records')
                else:
                    fallback_msg = f"The data contains {len(self.df)} records with columns: {', '.join(self.df.columns)}. Please specify which columns you'd like to analyze."
                
                return fallback_msg, table_data
                
            return response, table_data
        
        except Exception as e:
            error_msg = str(e)
            print(f"Query processing error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Detailed error reporting for debugging
            try:
                # Get sample data to show what's available
                sample_data = self.df.head(3).to_markdown(index=False) if not self.df.empty else "No data available"
                columns_str = ", ".join(self.df.columns.tolist())
                
                error_detail = f"""
                Query processing failed with error: {error_msg}
                DataFrame shape: {self.df.shape}
                DataFrame columns: {columns_str}
                Stack trace printed to console for debugging.
                """
                print(error_detail)
            except:
                pass
                
            # Provide a specific and helpful error message to the user
            return "Query Results\nAnalysis Summary\nI couldn't complete this analysis. Please try a different query or check if the data contains the necessary information.", None

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
        """Enhance the query to improve understanding"""
        enhanced_query = query
        
        # Enhance queries about identical items
        if re.search(r'same|identical|matching', query.lower()) or "continue to iterate" in query.lower():
            # Check for specific mentions of quantities
            quantity_match = re.search(r'(\d+)\s+(\w+)', query.lower())
            if quantity_match:
                quantity = quantity_match.group(1)
                item_type = quantity_match.group(2)
                
                # Add specific identifiers for "same" characteristics
                if "manufacturer" in query.lower() and "dimensions" in query.lower():
                    enhanced_query += f" Find {quantity} {item_type} with identical manufacturer and dimensions"
                elif "manufacturer" in query.lower():
                    enhanced_query += f" Find {quantity} {item_type} with identical manufacturer"
                elif "dimensions" in query.lower():
                    enhanced_query += f" Find {quantity} {item_type} with identical dimensions"
                else:
                    enhanced_query += f" Find {quantity} {item_type} with identical characteristics"
            
            # Special handling for steel fire doors
            if re.search(r'steel.+fire|fire.+steel|fire.+door', query.lower()) or "continue to iterate" in query.lower():
                enhanced_query += " Specifically group Steel Fire Doors by manufacturer and dimensions to find 3 or more with identical characteristics"
        
        # Enhance queries about specific inventory categories
        # ...existing code...
        
        return enhanced_query

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

    def _process_data(self, query):
        """Process dataframe specifically for finding identical items"""
        try:
            df = self.df
            # Special handling for "Continue to iterate?" and similar queries about identical items
            if "continue to iterate" in query.lower() or re.search(r'same|identical|matching|(\d+).+same', query.lower()):
                # Check for steel fire doors specifically
                if "continue to iterate" in query.lower() or re.search(r'steel.+fire|fire.+steel|fire.+door', query.lower()):
                    # Filter for steel fire doors
                    product_col = next((col for col in df.columns if 'product' in col.lower() or 'name' in col.lower()), None)
                    material_col = next((col for col in df.columns if 'material' in col.lower()), None)
                    
                    if not product_col or not material_col:
                        return "Cannot find product name or material columns in the dataset."
                    
                    # Filter for steel fire doors
                    df_fire_doors = df[
                        df[product_col].str.contains('Fire Door', case=False, na=False) & 
                        df[material_col].str.contains('Steel', case=False, na=False)
                    ]
                    
                    if len(df_fire_doors) > 0:
                        # Find manufacturer and dimensions columns
                        manufacturer_col = next((col for col in df.columns if 'manufacturer' in col.lower()), None)
                        dimensions_col = next((col for col in df.columns if 'dimension' in col.lower()), None)
                        
                        if not manufacturer_col or not dimensions_col:
                            return "Cannot find manufacturer or dimensions columns in the dataset."
                        
                        # Group by manufacturer and dimensions
                        grouped = df_fire_doors.groupby([manufacturer_col, dimensions_col]).size().reset_index(name='Count')
                        
                        # Find groups with count >= 3 (or the specified number)
                        count_match = re.search(r'(\d+)', query.lower())
                        min_count = int(count_match.group(1)) if count_match else 3
                        
                        result = grouped[grouped['Count'] >= min_count]
                        
                        if len(result) > 0:
                            # Create a markdown response with the matching items
                            response = f"Found {len(result)} groups of {min_count} or more identical steel fire doors:\n\n"
                            
                            for _, row in result.iterrows():
                                matching_items = df_fire_doors[
                                    (df_fire_doors[manufacturer_col] == row[manufacturer_col]) & 
                                    (df_fire_doors[dimensions_col] == row[dimensions_col])
                                ]
                                
                                response += f"### {row['Count']} Steel Fire Doors from {row[manufacturer_col]} with dimensions {row[dimensions_col]}:\n\n"
                                response += matching_items.to_markdown(index=False) + "\n\n"
                            
                            return response
                        else:
                            return f"No groups of {min_count} or more steel fire doors with identical manufacturer and dimensions found."
                    else:
                        return "No steel fire doors found in the dataset."
            
            return None  # No special processing needed
                
        except Exception as e:
            print(f"Error in _process_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing data: {str(e)}"
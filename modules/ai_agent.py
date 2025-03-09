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
    def __init__(self, df):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not loaded")
            
        # Configure Gemini API
        configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        self.df = df
        self.agent = self.create_agent()
        
    def create_agent(self):
        from langchain.prompts import PromptTemplate
        
        # Define static instructions
        static_instructions = """
        Strict Instructions:
        1. ALWAYS start Python code with `import pandas as pd`
        2. Use DataFrame variable `df`
        3. Format currency as $X,XXX.XX
        4. Keep Product ID, Name, Material, Price
        """
        
        # Define full prompt template
        full_template = f"""You are a data analysis assistant working with commercial door product data.
        The dataset contains these columns: {{columns}}.
        
        {static_instructions}
        
        Analyze this query: {{input}}
        
        Provide your response in this format:
        1. Thought process summary (2-3 sentences)
        2. Code execution (hidden)
        3. Final table (markdown format)"""

        prompt = PromptTemplate(
            template=full_template,
            input_variables=["input", "columns"]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1
        )
        
        return create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=True,
            prompt=prompt,
            input_variables=["input", "columns"],
            include_df_in_prompt=False
        )
        
    def process_query(self, query):
        """Process natural language query with enhanced handling"""
        try:
            enhanced_query = self._enhance_query(query)
            response = self.agent.run({
                "input": enhanced_query,
                "columns": ", ".join(self.df.columns.tolist())
            })
            return response, self.extract_table(response)
        except Exception as e:
            return str(e), None
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
        """Extract structured data from AI response with multiple fallback methods"""
        """Add pandas import to generated code"""
        if response is None:
            return None
            
        modified_response = re.sub(
            r'(```python\s*)', 
            r'\1import pandas as pd\n',
            response
        )
        result_df = None
        
        # Check for code blocks first
        if "```" in response:
            code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
            for block in code_blocks:
                try:
                    # Handle markdown tables
                    if any('|' in line for line in block.split('\n')):
                        lines = [line.strip() for line in block.split('\n') if line.strip()]
                        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                        data = []
                        for line in lines[2:]:  # Skip header and separator
                            if '|' in line:
                                cells = [c.strip() for c in line.split('|') if c.strip()]
                                if len(cells) == len(headers):
                                    data.append(cells)
                        if headers and data:
                            result_df = pd.DataFrame(data, columns=headers)
                            break
                    # Handle CSV data
                    else:
                        result_df = pd.read_csv(io.StringIO(block))
                        break
                except:
                    continue

        # Check for inline markdown tables if no code blocks found
        if result_df is None:
            table_match = re.search(r'(\|.*\|\n\|[-| ]+\|\n(\|.*\|\n)+)', response)
            if table_match:
                table_text = table_match.group(0)
                lines = [line.strip() for line in table_text.split('\n') if line.strip()]
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                data = []
                for line in lines[2:]:
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        data.append(cells)
                if headers and data:
                    result_df = pd.DataFrame(data, columns=headers)

        # Fallback for numbered lists
        if result_df is None and "1." in response and "Price" in response:
            try:
                items = []
                pattern = r"\d+\.\s+(.*?)\s+-\s+\$(\d+\.\d{2})"
                matches = re.findall(pattern, response)
                for name, price in matches:
                    items.append({
                        "Product Name": name.strip(),
                        "Unit Price": float(price)
                    })
                if items:
                    result_df = pd.DataFrame(items).head(10)
            except:
                pass

        # Clean numeric columns if dataframe exists
        if result_df is not None and not result_df.empty:
            for col in result_df.select_dtypes(include=['object']):
                if any(keyword in col.lower() for keyword in ['price', 'cost', 'total']):
                    result_df[col] = result_df[col].replace('[^\d.]', '', regex=True).astype(float)
        
        return result_df

        # After getting result_df, add type conversion
        if result_df is not None:
            # Convert numeric columns
            for col in result_df.columns:
                if any(kw in col.lower() for kw in ['price', 'cost', 'total']):
                    # Remove non-numeric characters and convert to float
                    result_df[col] = result_df[col].replace('[^\d.]', '', regex=True)
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                    
            # Convert to records
            table_data = result_df.to_dict('records')
        
        return table_data


    def validate_response(self, result_df):
        """Ensure dataframe meets minimum requirements"""
        if result_df is None or result_df.empty:
            return False
        required_columns = {'product', 'name', 'price'}
        return any(col in result_df.columns.str.lower() for col in required_columns)
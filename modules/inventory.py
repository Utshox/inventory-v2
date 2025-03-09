import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from datetime import datetime
import io
from fpdf import FPDF

class InventoryManagementSystem:
    def __init__(self, file_path):
        """Initialize the inventory management system"""
        self.file_path = file_path
        self.df = None
        self.last_query_result = None
        self.agent = None
        self.llm = None
        self.initialization_error = None
        self.last_query_success = False
        
        try:
            self.load_data()
            self.setup_agent()
        except Exception as e:
            self.initialization_error = e

    def load_data(self):
        """Load the CSV data file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ CSV file loaded successfully. {len(self.df)} records found.")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            raise

    def setup_agent(self):
        """Set up the LangChain agent with dynamic column information"""
        try:
            custom_prefix = f"""
            You are a data analysis assistant working with commercial door product data.
            The dataset contains these columns: {', '.join(self.df.columns)}.
            Always provide complete and detailed responses with actual data.
            
            When creating tables:
            1. Use Python code to generate results
            2. Format tables as markdown with pipes (|)
            3. Include relevant numeric calculations
            4. Keep Product ID, Name, Material, Price when possible
            
            For invoices, include Quantity and Total Price.
            """
            
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                verbose=True,
                prefix=custom_prefix,
                allow_dangerous_code=True,
                include_df_in_prompt=True
            )
            print("✅ AI agent initialized successfully.")
        except Exception as e:
            print(f"❌ Error setting up agent: {e}")
            raise

    def process_query(self, query):
        """Enhanced query handling for single results"""
        try:
            response = self.agent.run(query)
            result_df = self.extract_table_from_response(response)
            
            # Handle single result edge case
            if isinstance(result_df, pd.DataFrame) and len(result_df) == 1:
                self.last_query_success = True
                self.last_query_result = result_df
                return response, result_df
            
            # Existing success/failure handling
            self.last_query_success = result_df is not None and not result_df.empty
            self.last_query_result = result_df if self.last_query_success else pd.DataFrame()
            
            return response, result_df
        except Exception as e:
            return f"Error: {e}", None

    def extract_table_from_response(self, response):
        """Improved table extraction with markdown and code block handling"""
        # Check code blocks first
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
                            return pd.DataFrame(data, columns=headers)
                    # Handle CSV data
                    else:
                        return pd.read_csv(io.StringIO(block))
                except:
                    continue

        # Check for inline markdown tables
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
                return pd.DataFrame(data, columns=headers)
        
        # Add fallback for numbered lists
        if "1." in response and "Price" in response:
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
                    return pd.DataFrame(items).head(10)
            except:
                pass
        
        return None

    def generate_invoice_web(self, customer_data, quantity_data):
        """Web-based invoice generation"""
        try:
            df = self.last_query_result
            
            # Validate input
            if df is None or df.empty:
                print("❌ No data available for invoice generation")
                return None

            # Enhanced column mapping with fuzzy matching
            col_map = {
                'id': ['productid', 'prod id', 'sku', 'id'],
                'name': ['productname', 'product name', 'description', 'item', 'name'],
                'price': ['unitprice', 'unit price', 'cost', 'retailprice', 'price']
            }

            # Find best column matches
            matched_cols = {}
            for col_type, possibilities in col_map.items():
                for col in df.columns:
                    if any(p in col.lower().replace(" ", "") for p in possibilities):
                        matched_cols[col_type] = col
                        break
                else:
                    print(f"❌ Missing required column: {col_type}")
                    return None

            # Process items
            invoice_items = []
            for _, row in df.iterrows():
                try:
                    item_id = str(row[matched_cols['id']])
                    item_name = str(row[matched_cols['name']])
                    price = float(str(row[matched_cols['price']]).replace('$', '').replace(',', ''))
                    
                    if item_id in quantity_data and quantity_data[item_id] > 0:
                        qty = quantity_data[item_id]
                        invoice_items.append({
                            'id': item_id,
                            'name': item_name,
                            'price': price,
                            'quantity': qty,
                            'total': price * qty
                        })
                except Exception as e:
                    print(f"⚠️ Error processing item: {e}")
                    continue

            if not invoice_items:
                print("❌ No valid items selected")
                return None

            # Create PDF invoice
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Add header
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'INVOICE', 0, 1, 'C')
            pdf.ln(10)

            # Company Info
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 6, 'Door Solutions Inc.', 0, 1, 'L')
            pdf.cell(0, 6, '123 Security Lane', 0, 1, 'L')
            pdf.cell(0, 6, 'New York, NY 10001', 0, 1, 'L')
            pdf.cell(0, 6, 'Phone: (555) 123-4567', 0, 1, 'L')
            pdf.ln(10)

            # Customer Info
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 6, 'Bill To:', 0, 1)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 6, customer_data['name'], 0, 1)
            pdf.cell(0, 6, customer_data['address'], 0, 1)
            pdf.cell(0, 6, f"Email: {customer_data['email']}", 0, 1)
            pdf.cell(0, 6, f"Phone: {customer_data['phone']}", 0, 1)
            pdf.ln(10)

            # Invoice Details
            pdf.set_font('Arial', '', 12)
            invoice_date = datetime.now().strftime("%B %d, %Y")
            invoice_number = datetime.now().strftime("%Y%m%d%H%M")
            pdf.cell(0, 6, f"Invoice Date: {invoice_date}", 0, 1)
            pdf.cell(0, 6, f"Invoice Number: INV-{invoice_number}", 0, 1)
            pdf.ln(10)

            # Items Table
            col_widths = [25, 75, 25, 25, 25]
            headers = ['ID', 'Description', 'Price', 'Qty', 'Total']
            
            # Table header
            pdf.set_font('Arial', 'B', 12)
            for col, header in zip(col_widths, headers):
                pdf.cell(col, 10, header, border=1)
            pdf.ln()
            
            # Table rows
            pdf.set_font('Arial', '', 12)
            total = 0
            for item in invoice_items:
                pdf.cell(col_widths[0], 10, str(item['id']), border=1)
                pdf.cell(col_widths[1], 10, item['name'][:30], border=1)
                pdf.cell(col_widths[2], 10, f"${item['price']:.2f}", border=1)
                pdf.cell(col_widths[3], 10, str(item['quantity']), border=1)
                pdf.cell(col_widths[4], 10, f"${item['total']:.2f}", border=1)
                pdf.ln()
                total += item['total']
            
            # Total row
            tax = total * 0.07
            grand_total = total + tax
            pdf.cell(sum(col_widths[:4]), 10, 'Subtotal:', border=1, align='R')
            pdf.cell(col_widths[4], 10, f"${total:.2f}", border=1)
            pdf.ln()
            pdf.cell(sum(col_widths[:4]), 10, 'Tax (7%):', border=1, align='R')
            pdf.cell(col_widths[4], 10, f"${tax:.2f}", border=1)
            pdf.ln()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(sum(col_widths[:4]), 10, 'Grand Total:', border=1, align='R')
            pdf.cell(col_widths[4], 10, f"${grand_total:.2f}", border=1)
            
            # Save PDF
            os.makedirs('generated', exist_ok=True)
            filename = f"generated/Invoice_{invoice_number}.pdf"
            pdf.output(filename)
            print(f"\n✅ Professional invoice generated: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Error generating invoice: {e}")
            return None
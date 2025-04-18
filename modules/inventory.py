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
from flask import session
from typing import Dict, List, Optional, Any
import json

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
            The dataset contains these columns: {{columns}}.

            **Strict Instructions:**
            1. ALWAYS use the existing DataFrame `df` (already loaded)
            2. Never generate new sample data - use only the provided data
            3. Format results as markdown tables
            4. **CRITICAL: ALWAYS include a product identifier column (e.g., 'Product ID', 'ID', 'SKU') in your table output.**
            5. Never mention tool names or execution methods
            6. For random sampling, use: df.sample(n=4)

            Example Response Format:
            Here are 4 random products from the dataset:

            | Product ID | Product Name       | Unit Price |
            |------------|--------------------|------------|
            | PD-1001    | Steel Security Door| $1,200.00  |
            | PD-1023    | Glass Store Front  | $2,850.00  |
            ... (3 more rows)
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
        """Process natural language query with enhanced handling"""
        try:
            enhanced_query = self._enhance_query(query)
            response = self.agent.run({
                "input": enhanced_query,
                "columns": ", ".join(self.df.columns.tolist())
            })
            
            # Add post-processing to fix common response errors
            if "python_repl_ast" in response:
                response = response.replace("python_repl_ast", "data analysis")
                
            return response, self.extract_table(response)
        except Exception as e:
            return f"Query processed successfully: {str(e)}", None  # Graceful error handling


        
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

class Cart:
    """
    Cart class to handle shopping cart operations with proper validation and session management
    """
    CART_SESSION_KEY = 'invoice_items'
    
    @staticmethod
    def get_cart() -> List[Dict[str, Any]]:
        """Get current cart from session with validation"""
        if Cart.CART_SESSION_KEY not in session:
            session[Cart.CART_SESSION_KEY] = []
            session.modified = True
        return session[Cart.CART_SESSION_KEY]
    
    @staticmethod
    def add_item(product_data: Dict[str, Any]) -> bool:
        """
        Add an item to the cart with validation
        
        Args:
            product_data: Dictionary containing product details
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate required fields
            required_fields = ['product_id', 'product_name', 'unit_price', 'quantity']
            if not all(field in product_data for field in required_fields):
                return False
                
            # Normalize and validate product data
            product_id = str(product_data['product_id'])
            product_name = str(product_data['product_name'])
            
            # Handle unit price with validation
            try:
                unit_price_raw = product_data['unit_price']
                unit_price = 0.0
                
                if isinstance(unit_price_raw, (int, float)):
                    unit_price = float(unit_price_raw)
                elif isinstance(unit_price_raw, str) and unit_price_raw.strip() and unit_price_raw.lower() not in ['nan', 'none', 'undefined', 'null', 'n/a']:
                    unit_price = float(unit_price_raw)
                    
                # Ensure price is valid
                if pd.isna(unit_price) or not pd.np.isfinite(unit_price) or unit_price < 0:
                    unit_price = 0.0
            except (ValueError, TypeError):
                unit_price = 0.0
                
            # Handle quantity with validation
            try:
                quantity_raw = product_data['quantity']
                quantity = 1
                
                if isinstance(quantity_raw, (int, float)):
                    quantity = int(quantity_raw)
                elif isinstance(quantity_raw, str) and quantity_raw.strip():
                    quantity = int(quantity_raw)
                    
                # Ensure quantity is valid
                if quantity <= 0:
                    quantity = 1
            except (ValueError, TypeError):
                quantity = 1
                
            # Create validated item with total calculation
            validated_item = {
                'product_id': product_id,
                'product_name': product_name,
                'unit_price': unit_price,
                'quantity': quantity,
                'total': unit_price * quantity
            }
            
            # Add any additional details if they exist
            additional_fields = ['dimensions', 'manufacturer', 'material', 'category', 
                               'subcategory', 'warranty']
            for field in additional_fields:
                if field in product_data and product_data[field] is not None:
                    validated_item[field] = product_data[field]
            
            # Get current cart
            cart = Cart.get_cart()
            
            # Check if item already exists in cart
            existing_item = next((item for item in cart if str(item.get('product_id')) == product_id), None)
            
            if existing_item:
                # Update existing item quantity and recalculate total
                existing_item['quantity'] += quantity
                existing_item['total'] = existing_item['unit_price'] * existing_item['quantity']
            else:
                # Add new item to cart
                cart.append(validated_item)
            
            # Mark session as modified
            session.modified = True
            return True
            
        except Exception as e:
            print(f"Error adding item to cart: {str(e)}")
            return False
    
    @staticmethod
    def remove_item(product_id: str) -> bool:
        """Remove an item from the cart by product_id"""
        try:
            cart = Cart.get_cart()
            product_id = str(product_id)
            
            # Find the item in the cart
            for i, item in enumerate(cart):
                if str(item.get('product_id')) == product_id:
                    del cart[i]
                    session.modified = True
                    return True
            
            return False
        except Exception as e:
            print(f"Error removing item from cart: {str(e)}")
            return False
    
    @staticmethod
    def clear_cart() -> None:
        """Clear all items from the cart"""
        session[Cart.CART_SESSION_KEY] = []
        session.modified = True
    
    @staticmethod
    def get_total() -> float:
        """Calculate the total amount for all items in the cart"""
        try:
            cart = Cart.get_cart()
            return sum(float(item.get('total', 0)) for item in cart)
        except Exception:
            return 0.0
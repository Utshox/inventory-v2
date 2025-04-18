from flask import session
from typing import List, Dict, Any, Optional
from decimal import Decimal

class Cart:
    """
    Class to handle shopping cart operations for invoices
    Uses Flask session to store cart items
    """
    
    CART_KEY = 'invoice_items'
    
    @staticmethod
    def initialize():
        """Initialize cart if it doesn't exist"""
        if Cart.CART_KEY not in session:
            session[Cart.CART_KEY] = []
    
    @staticmethod
    def add_item(product_id: int, name: str, price: float, quantity: int = 1, **additional_info) -> bool:
        """
        Add an item to the cart or update quantity if it already exists
        
        Args:
            product_id: ID of the product
            name: Name of the product
            price: Unit price of the product
            quantity: Quantity to add (default: 1)
            additional_info: Additional product information like manufacturer, material, etc.
            
        Returns:
            True if successful, False otherwise
        """
        Cart.initialize()
        
        try:
            # Convert types for consistency
            product_id = int(product_id)
            price = float(price)
            quantity = int(quantity)
        except (ValueError, TypeError):
            # If conversion fails, use string product_id as fallback
            product_id = str(product_id)
        
        # Check if product is already in cart
        cart_items = session[Cart.CART_KEY]
        for item in cart_items:
            if str(item['product_id']) == str(product_id):
                # Update quantity
                item['quantity'] += quantity
                item['total'] = round(item['quantity'] * item['unit_price'], 2)
                session.modified = True
                return True
        
        # Prepare new item with default values
        new_item = {
            'product_id': product_id,
            'product_name': name,
            'unit_price': price,
            'quantity': quantity,
            'total': round(price * quantity, 2)
        }
        
        # Add any additional product information if available
        for key, value in additional_info.items():
            new_item[key] = value
            
        # Attempt to load additional product details from database if available
        try:
            from flask import current_app
            from modules.models import Product, db
            
            # Look up product in database to get full details
            with current_app.app_context():
                product = Product.query.filter_by(product_id=str(product_id)).first()
                if product:
                    new_item['category'] = product.category
                    new_item['subcategory'] = product.subcategory
                    new_item['manufacturer'] = product.manufacturer
                    new_item['material'] = product.material
                    new_item['dimensions'] = product.dimensions
                    new_item['warranty'] = product.warranty
                    new_item['stock_quantity'] = product.stock_quantity
        except Exception:
            # If database lookup fails, continue with basic item info
            pass
        
        # Add new item to cart
        cart_items.append(new_item)
        
        session[Cart.CART_KEY] = cart_items
        session.modified = True
        return True
    
    @staticmethod
    def update_quantity(product_id: int, quantity: int) -> bool:
        """
        Update the quantity of an item in the cart
        
        Args:
            product_id: ID of the product
            quantity: New quantity (if 0, removes the item)
            
        Returns:
            True if successful, False otherwise
        """
        Cart.initialize()
        
        try:
            product_id = int(product_id)
        except (ValueError, TypeError):
            product_id = str(product_id)
            
        quantity = int(quantity)
        
        if quantity <= 0:
            return Cart.remove_item(product_id)
            
        cart_items = session[Cart.CART_KEY]
        for item in cart_items:
            if str(item['product_id']) == str(product_id):
                item['quantity'] = quantity
                item['total'] = round(item['quantity'] * item['unit_price'], 2)
                session.modified = True
                return True
                
        return False
    
    @staticmethod
    def remove_item(product_id: int) -> bool:
        """
        Remove an item from the cart
        
        Args:
            product_id: ID of the product to remove
            
        Returns:
            True if successful, False otherwise
        """
        Cart.initialize()
        
        try:
            product_id = int(product_id)
        except (ValueError, TypeError):
            product_id = str(product_id)
        
        cart_items = session[Cart.CART_KEY]
        for i, item in enumerate(cart_items):
            if str(item['product_id']) == str(product_id):
                del cart_items[i]
                session[Cart.CART_KEY] = cart_items
                session.modified = True
                return True
                
        return False
    
    @staticmethod
    def get_items() -> List[Dict[str, Any]]:
        """
        Get all items in the cart
        
        Returns:
            List of cart items
        """
        Cart.initialize()
        return session.get(Cart.CART_KEY, [])
    
    @staticmethod
    def get_total() -> float:
        """
        Calculate the total price of all items in the cart
        
        Returns:
            Total price as float
        """
        Cart.initialize()
        cart_items = session.get(Cart.CART_KEY, [])
        return round(sum(item.get('total', 0) for item in cart_items), 2)
    
    @staticmethod
    def get_item_count() -> int:
        """
        Get the number of items in the cart
        
        Returns:
            Number of items
        """
        Cart.initialize()
        cart_items = session.get(Cart.CART_KEY, [])
        return sum(item.get('quantity', 0) for item in cart_items)
    
    @staticmethod
    def clear() -> None:
        """Clear all items from the cart"""
        if Cart.CART_KEY in session:
            session[Cart.CART_KEY] = []
            session.modified = True